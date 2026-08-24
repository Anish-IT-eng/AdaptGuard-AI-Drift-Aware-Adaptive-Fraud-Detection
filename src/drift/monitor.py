"""
AdaptGuard AI — Unified Drift Orchestrator
Combines the two parallel monitoring channels into one cohesive interface:

  CHANNEL 1 — Data Drift (label-free):
    PSI + KS + MMD on feature distributions
    Fires immediately — no label dependency

  CHANNEL 2 — Performance Drift (label-required):
    ADWIN + Page-Hinkley on prediction errors
    Fires only after delayed labels are confirmed

The orchestrator exposes a single step() interface so the controller
never has to manage both monitors separately.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.drift.data_monitor import DataDriftMonitor, DataDriftResult
from src.drift.perf_monitor import PerformanceDriftMonitor, PerfDriftResult
from src.drift.severity import SeverityEstimator, SeverityAssessment, SeverityLevel

log = get_logger("drift.monitor")


@dataclass
class DriftStep:
    """
    Result from one orchestrator step.
    Combines both channels into a single snapshot.
    """
    timestamp:           pd.Timestamp
    data_result:         Optional[DataDriftResult]   = None
    perf_result:         Optional[PerfDriftResult]   = None
    assessment:          Optional[SeverityAssessment] = None
    # Convenience flags
    data_drift_detected:  bool  = False
    perf_drift_detected:  bool  = False
    any_drift_detected:   bool  = False
    severity_level:       str   = "NONE"
    severity_score:       float = 0.0


class DriftOrchestrator:
    """
    Single entry point for all drift monitoring in AdaptGuard AI.

    Usage pattern inside the streaming loop:

        # On every new transaction batch (label-free):
        step = orchestrator.update_data(current_batch_df, timestamp)

        # When a confirmed label arrives from the delayed-label buffer:
        step = orchestrator.update_performance(y_true, y_pred, timestamp)

        # Query current overall severity at any time:
        assessment = orchestrator.current_assessment
    """

    def __init__(
        self,
        feature_cols:    list[str],
        reference_df:    pd.DataFrame,
        cfg:             Optional[dict] = None,
    ):
        """
        Args:
            feature_cols:  Feature columns monitored for data drift.
            reference_df:  Training-window DataFrame used to set reference distributions.
            cfg:           Config dict (loads default if None).
        """
        self.cfg  = cfg or load_config()
        self.feature_cols = feature_cols

        # Sub-monitors
        self.data_monitor = DataDriftMonitor(cfg=self.cfg)
        self.perf_monitor = PerformanceDriftMonitor(cfg=self.cfg)
        self.severity_est = SeverityEstimator()

        # Initialize data monitor reference distributions
        self.data_monitor.initialize(reference_df, feature_cols)
        log.info(
            f"DriftOrchestrator initialized | "
            f"features={len(feature_cols)} | "
            f"reference_rows={len(reference_df):,}"
        )

        # State
        self._last_data_result:   Optional[DataDriftResult]   = None
        self._last_perf_result:   Optional[PerfDriftResult]   = None
        self._last_assessment:    Optional[SeverityAssessment] = None
        self.step_history: list[DriftStep] = []

    # ------------------------------------------------------------------
    # Channel 1 — Data Drift (label-free, call on each batch)
    # ------------------------------------------------------------------

    def update_data(
        self,
        current_df: pd.DataFrame,
        timestamp:  pd.Timestamp,
        run_mmd:    bool = False,
    ) -> DriftStep:
        """
        Run one data-drift monitoring cycle (label-free).

        Call this with the current sliding window of transactions.
        Does NOT require fraud labels — fires immediately.

        Args:
            current_df: Current window of transactions (feature columns required).
            timestamp:  Logical timestamp for this batch.
            run_mmd:    Whether to run the more expensive MMD test.

        Returns:
            DriftStep with data channel results and updated severity.
        """
        data_result = self.data_monitor.monitor(
            current_df = current_df,
            timestamp  = timestamp,
            run_mmd    = run_mmd,
        )
        self._last_data_result = data_result

        # Re-assess severity with latest data signal
        assessment = self.severity_est.assess(
            data_result = data_result,
            perf_result = self._last_perf_result,
        )
        self._last_assessment = assessment

        step = DriftStep(
            timestamp            = timestamp,
            data_result          = data_result,
            perf_result          = self._last_perf_result,
            assessment           = assessment,
            data_drift_detected  = data_result.drift_detected,
            perf_drift_detected  = (
                self._last_perf_result.detection_event
                if self._last_perf_result else False
            ),
            any_drift_detected   = (
                data_result.drift_detected
                or (self._last_perf_result.detection_event if self._last_perf_result else False)
            ),
            severity_level  = str(assessment.level),
            severity_score  = assessment.score,
        )
        self.step_history.append(step)

        if data_result.drift_detected:
            log.info(
                f"[Orchestrator] DATA DRIFT | "
                f"Severity={assessment.level} ({assessment.score:.3f}) | "
                f"{data_result.summary}"
            )

        return step

    # ------------------------------------------------------------------
    # Channel 2 — Performance Drift (label-required, call per confirmed label)
    # ------------------------------------------------------------------

    def update_performance(
        self,
        y_true:    int,
        y_pred:    int,
        timestamp: pd.Timestamp,
    ) -> DriftStep:
        """
        Update the performance drift monitor with one confirmed label.

        Call this ONLY when a confirmed label arrives from the delayed-label buffer.
        Requires: y_true = confirmed ground-truth label after delay period.

        Args:
            y_true:    Confirmed fraud label (0 or 1).
            y_pred:    The prediction made at transaction time.
            timestamp: Logical timestamp of the confirmed label release.

        Returns:
            DriftStep with performance channel results and updated severity.
        """
        perf_result = self.perf_monitor.update(y_true, y_pred)
        self._last_perf_result = perf_result

        # Re-assess severity with latest performance signal
        assessment = self.severity_est.assess(
            data_result = self._last_data_result,
            perf_result = perf_result,
        )
        self._last_assessment = assessment

        step = DriftStep(
            timestamp            = timestamp,
            data_result          = self._last_data_result,
            perf_result          = perf_result,
            assessment           = assessment,
            data_drift_detected  = (
                self._last_data_result.drift_detected
                if self._last_data_result else False
            ),
            perf_drift_detected  = perf_result.detection_event,
            any_drift_detected   = (
                perf_result.detection_event
                or (self._last_data_result.drift_detected if self._last_data_result else False)
            ),
            severity_level  = str(assessment.level),
            severity_score  = assessment.score,
        )
        self.step_history.append(step)

        if perf_result.detection_event:
            log.warning(
                f"[Orchestrator] PERF DRIFT | "
                f"Detector={perf_result.detector_fired} | "
                f"ErrorRate={perf_result.current_error_rate:.4f} | "
                f"Severity={assessment.level} ({assessment.score:.3f})"
            )

        return step

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def current_assessment(self) -> Optional[SeverityAssessment]:
        """Most recent severity assessment (None if no steps yet)."""
        return self._last_assessment

    @property
    def current_severity_level(self) -> SeverityLevel:
        """Current severity level enum value."""
        if self._last_assessment:
            return self._last_assessment.level
        return SeverityLevel.NONE

    @property
    def should_adapt(self) -> bool:
        """
        True if current severity meets the minimum threshold for adaptation.
        Reads threshold from config: adaptation.min_severity_to_adapt
        """
        if self._last_assessment is None:
            return False
        min_sev_str = self.cfg["adaptation"].get("min_severity_to_adapt", "MEDIUM")
        min_level   = SeverityLevel[min_sev_str]
        return self._last_assessment.level >= min_level

    def reset_detectors(self) -> None:
        """
        Reset performance monitors after a successful adaptation.
        Prevents the same drift signal from re-triggering immediately.
        Data drift monitor reference is NOT reset (it tracks the environment,
        not the model's response).
        """
        self.perf_monitor.reset_detectors()
        log.info("[Orchestrator] Performance detectors reset after adaptation.")

    def get_status_dict(self) -> dict:
        """Return a serializable status snapshot for API/dashboard consumption."""
        a = self._last_assessment
        d = self._last_data_result
        p = self._last_perf_result
        return {
            "severity_level":        str(a.level)          if a else "NONE",
            "severity_score":        a.score                if a else 0.0,
            "adwin_signal":          a.adwin_signal         if a else False,
            "ph_signal":             a.ph_signal            if a else False,
            "max_psi":               a.max_psi              if a else 0.0,
            "error_trend":           a.error_trend          if a else 0.0,
            "data_drift_detected":   d.drift_detected       if d else False,
            "perf_drift_detected":   p.detection_event      if p else False,
            "current_error_rate":    p.current_error_rate   if p else 0.0,
            "n_perf_samples":        p.n_samples_processed  if p else 0,
            "n_data_monitor_cycles": len(self.data_monitor.history),
        }

    def summary(self) -> str:
        """Human-readable summary of the current drift state."""
        s = self.get_status_dict()
        return (
            f"Severity={s['severity_level']} ({s['severity_score']:.3f}) | "
            f"ADWIN={s['adwin_signal']} | PH={s['ph_signal']} | "
            f"MaxPSI={s['max_psi']:.4f} | "
            f"ErrorRate={s['current_error_rate']:.4f} | "
            f"ErrorTrend={s['error_trend']:.4f}"
        )

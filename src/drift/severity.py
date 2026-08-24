"""
AdaptGuard AI — Drift Severity Estimator
Combines signals from both monitoring channels into a single severity level.

Severity Levels: NONE → LOW → MEDIUM → HIGH → CRITICAL

IMPORTANT: Exact thresholds must be determined experimentally.
Values in config.yaml are starting points only and must be calibrated
on the actual dataset before drawing research conclusions.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from src.drift.data_monitor import DataDriftResult
from src.drift.perf_monitor import PerfDriftResult
from src.utils.logger import get_logger

log = get_logger("drift.severity")


class SeverityLevel(IntEnum):
    NONE     = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

    def __str__(self):
        return self.name


@dataclass
class SeverityAssessment:
    """Full severity assessment from combined monitoring signals."""
    level:              SeverityLevel
    score:              float           # 0.0 – 1.0 composite score
    adwin_signal:       bool
    ph_signal:          bool
    max_psi:            float
    error_rate:         float
    error_trend:        float
    mmd_score:          Optional[float]
    n_features_alert:   int
    recommended_action: str
    explanation:        str


class SeverityEstimator:
    """
    Aggregates signals from the data drift monitor (label-free)
    and the performance drift monitor (label-required) into a
    calibrated severity assessment.

    Decision logic (conceptual — thresholds to be calibrated experimentally):

    NONE:     No signals from either channel
    LOW:      Minor PSI elevation, no ADWIN signal, flat error trend
    MEDIUM:   Moderate PSI + ADWIN signal OR significant error trend
    HIGH:     Strong PSI + ADWIN + increasing error rate
    CRITICAL: All signals firing + error rate critically elevated
    """

    def __init__(
        self,
        # Thresholds below are STARTING POINTS — calibrate experimentally
        psi_low:       float = 0.10,
        psi_medium:    float = 0.15,
        psi_high:      float = 0.20,
        psi_critical:  float = 0.30,
        error_low:     float = 0.01,
        error_medium:  float = 0.03,
        error_high:    float = 0.05,
        error_critical: float = 0.10,
        trend_threshold: float = 0.02,
    ):
        self.psi_low        = psi_low
        self.psi_medium     = psi_medium
        self.psi_high       = psi_high
        self.psi_critical   = psi_critical
        self.error_low      = error_low
        self.error_medium   = error_medium
        self.error_high     = error_high
        self.error_critical = error_critical
        self.trend_threshold = trend_threshold

    def assess(
        self,
        data_result:  Optional[DataDriftResult],
        perf_result:  Optional[PerfDriftResult],
    ) -> SeverityAssessment:
        """
        Combine signals from both monitors into a severity level.

        Either result can be None (e.g., if no confirmed labels yet).
        """
        # Extract signals
        adwin_signal   = perf_result.adwin_detected     if perf_result else False
        ph_signal      = perf_result.ph_detected        if perf_result else False
        error_rate     = perf_result.current_error_rate if perf_result else 0.0
        error_trend    = perf_result.error_trend        if perf_result else 0.0
        max_psi        = data_result.max_psi            if data_result else 0.0
        n_alert        = data_result.n_features_alert   if data_result else 0
        mmd_score      = data_result.mmd_score          if data_result else None

        # Composite score (weighted combination)
        psi_component    = min(max_psi / self.psi_critical, 1.0)
        error_component  = min(error_rate / self.error_critical, 1.0)
        adwin_component  = 1.0 if adwin_signal else 0.0
        ph_component     = 0.5 if ph_signal else 0.0
        trend_component  = min(abs(error_trend) / 0.10, 1.0)

        score = (
            0.30 * psi_component
            + 0.30 * error_component
            + 0.20 * adwin_component
            + 0.10 * ph_component
            + 0.10 * trend_component
        )

        # Rule-based severity classification
        if error_rate >= self.error_critical and adwin_signal and max_psi >= self.psi_critical:
            level = SeverityLevel.CRITICAL
        elif adwin_signal and error_rate >= self.error_high and max_psi >= self.psi_high:
            level = SeverityLevel.HIGH
        elif (adwin_signal or ph_signal) and (error_rate >= self.error_medium or max_psi >= self.psi_medium):
            level = SeverityLevel.MEDIUM
        elif max_psi >= self.psi_low or error_rate >= self.error_low:
            level = SeverityLevel.LOW
        else:
            level = SeverityLevel.NONE

        # Recommended action
        actions = {
            SeverityLevel.NONE:     "Continue monitoring. No action required.",
            SeverityLevel.LOW:      "Increase monitoring frequency. Collect evidence.",
            SeverityLevel.MEDIUM:   "Collect recent evidence. Prepare candidate model.",
            SeverityLevel.HIGH:     "Train candidate model. Run validation gate.",
            SeverityLevel.CRITICAL: "Emergency adaptation. Immediate candidate training and validation.",
        }

        explanation = (
            f"PSI={max_psi:.4f} ({n_alert} features alert) | "
            f"ADWIN={'fired' if adwin_signal else 'stable'} | "
            f"PH={'fired' if ph_signal else 'stable'} | "
            f"ErrorRate={error_rate:.4f} | "
            f"ErrorTrend={error_trend:+.4f} | "
            f"Score={score:.3f}"
        )

        assessment = SeverityAssessment(
            level              = level,
            score              = float(score),
            adwin_signal       = adwin_signal,
            ph_signal          = ph_signal,
            max_psi            = max_psi,
            error_rate         = error_rate,
            error_trend        = error_trend,
            mmd_score          = mmd_score,
            n_features_alert   = n_alert,
            recommended_action = actions[level],
            explanation        = explanation,
        )

        if level >= SeverityLevel.HIGH:
            log.warning(f"[Severity] {level.name}: {explanation}")
        else:
            log.debug(f"[Severity] {level.name}: {explanation}")

        return assessment

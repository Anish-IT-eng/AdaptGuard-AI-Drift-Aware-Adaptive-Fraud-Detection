"""
AdaptGuard AI — Adaptive Controller
The core research component of the project.

Coordinates: drift monitoring → severity assessment → candidate training
             → validation gate → promotion or rejection → post-deployment monitoring
             → rollback if necessary

Key distinction from baselines:
- Static XGBoost:       Never updates
- Periodic Retraining:  Updates on calendar schedule regardless of drift
- Always-Online:        Updates on every confirmed label regardless of drift
- AdaptGuard AI:        Updates ONLY when drift evidence warrants it,
                        validates before promoting, monitors after, rolls back if needed
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Any, Callable

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.drift.data_monitor import DataDriftMonitor, DataDriftResult
from src.drift.perf_monitor import PerformanceDriftMonitor, PerfDriftResult
from src.drift.severity import SeverityEstimator, SeverityLevel, SeverityAssessment
from src.adaptation.delayed_labels import DelayedLabelBuffer, PendingLabel
from src.adaptation.candidate import CandidateTrainer, ValidationGate, ValidationResult
from src.models.registry import ModelRegistry
from src.evaluation.metrics import evaluate_predictions, RollingMetrics

log = get_logger("adaptation.controller")


@dataclass
class AdaptationEvent:
    """Records one adaptation cycle."""
    timestamp:        str
    severity_level:   str
    severity_score:   float
    action:           str           # "adapted", "rejected", "monitoring", "rollback"
    production_v:     int
    candidate_v:      Optional[int] = None
    validation_result: Optional[str] = None


class AdaptiveController:
    """
    AdaptGuard AI Core Adaptive Controller.

    Orchestrates the full adaptation lifecycle:
    1. Receive confirmed labels from the delayed-label buffer
    2. Update performance drift monitor (ADWIN/PH)
    3. Trigger severity assessment (data + performance signals)
    4. If severity >= threshold: train candidate model
    5. Run validation gate (Champion-Challenger)
    6. Promote candidate if gate passes, reject otherwise
    7. Monitor post-deployment performance
    8. Rollback if post-deployment degradation detected
    """

    def __init__(
        self,
        production_model:     Any,
        model_factory:        Callable,
        registry:             ModelRegistry,
        feature_cols:         list[str],
        reference_df:         pd.DataFrame,
        cfg:                  Optional[dict] = None,
        label_delay_days:     int = 3,
        ablation_flags:       Optional[dict] = None,
    ):
        """
        Args:
            production_model:  Initial production model.
            model_factory:     Callable that returns a new unfitted model.
            registry:          ModelRegistry instance.
            feature_cols:      Feature column names.
            reference_df:      Initial training data (for drift reference).
            cfg:               Config dict (loads default if None).
            label_delay_days:  Label delay for delayed-label buffer.
            ablation_flags:    Dict controlling which components are active.
                               Keys: use_drift_detection, use_severity,
                                     use_delayed_labels, use_validation_gate,
                                     use_rollback
                               All default True for full AdaptGuard AI.
        """
        self.cfg          = cfg or load_config()
        self.feature_cols = feature_cols
        self.registry     = registry
        self.model_factory = model_factory

        # Ablation flags (all True = full AdaptGuard AI)
        self.flags = {
            "use_drift_detection": True,
            "use_severity":        True,
            "use_delayed_labels":  True,
            "use_validation_gate": True,
            "use_rollback":        True,
        }
        if ablation_flags:
            self.flags.update(ablation_flags)

        # Sub-components
        self.data_monitor  = DataDriftMonitor(cfg=self.cfg)
        self.perf_monitor  = PerformanceDriftMonitor(cfg=self.cfg)
        self.severity_est  = SeverityEstimator()
        self.label_buffer  = DelayedLabelBuffer(
            delay_days=label_delay_days if self.flags["use_delayed_labels"] else 0
        )

        adapt_cfg = self.cfg["adaptation"]
        self.validation_gate = ValidationGate(
            prauc_improvement  = adapt_cfg["gate"]["prauc_improvement"],
            recall_floor       = adapt_cfg["gate"]["recall_floor"],
            fpr_ceiling        = adapt_cfg["gate"]["fpr_ceiling"],
            latency_ceiling_ms = adapt_cfg["gate"]["latency_ceiling_ms"],
        )
        self.candidate_trainer = CandidateTrainer(model_factory=model_factory)

        # Thresholds
        severity_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        min_sev_str  = adapt_cfg.get("min_severity_to_adapt", "MEDIUM")
        self.min_severity_to_adapt = SeverityLevel[min_sev_str]
        self.candidate_window_days = adapt_cfg["candidate_window_days"]
        self.validation_window_days = adapt_cfg["validation_window_days"]

        # Initialize data drift monitor with reference
        self.data_monitor.initialize(reference_df, feature_cols)

        # State
        self._confirmed_buffer: list[PendingLabel] = []   # Confirmed label store
        self._last_assessment: Optional[SeverityAssessment] = None
        self._last_data_result: Optional[DataDriftResult]   = None
        self._rolling = RollingMetrics(window_size=1000)

        self.adaptation_events:  list[AdaptationEvent] = []
        self.adaptation_count    = 0
        self.rejection_count     = 0

        # Post-deployment monitoring
        self._post_deploy_buffer: list = []
        self._post_deploy_active = False

        log.info(
            f"AdaptiveController initialized | "
            f"flags={self.flags} | "
            f"min_severity={self.min_severity_to_adapt} | "
            f"delay={label_delay_days}d"
        )

    # ------------------------------------------------------------------
    # Main streaming entry point
    # ------------------------------------------------------------------

    def process_transaction(
        self,
        row:       pd.Series,
        timestamp: pd.Timestamp,
    ) -> tuple[int, float]:
        """
        Process a single transaction through the adaptive system.

        Steps:
        1. Predict with current production model (before label known)
        2. Store in delayed-label buffer
        3. Release any confirmed labels from buffer
        4. Process confirmed labels (drift detection, metrics, adaptation)
        5. Monitor data drift on current window (label-free)

        Returns:
            (binary_prediction, fraud_probability)
        """
        # --- Get current production model ---
        try:
            prod_model, prod_record = self.registry.get_production_model()
        except RuntimeError:
            raise RuntimeError("No production model in registry.")

        # --- Predict ---
        X_row  = pd.DataFrame([row[self.feature_cols]])
        proba  = prod_model.predict_proba(X_row)[:, 1]
        prob   = float(proba[0])
        pred   = int(prob >= 0.5)

        # --- Store in label buffer ---
        self.label_buffer.store(
            transaction_id = int(row.get("TRANSACTION_ID", 0)),
            tx_datetime    = timestamp,
            y_true         = int(row["TX_FRAUD"]),
            y_pred         = pred,
            y_prob         = prob,
            features       = row[self.feature_cols],
        )

        # --- Release confirmed labels ---
        confirmed = self.label_buffer.release(current_time=timestamp)
        for pending in confirmed:
            self._process_confirmed_label(pending, prod_model)

        return pred, prob

    def _process_confirmed_label(
        self,
        pending:    PendingLabel,
        prod_model: Any,
    ) -> None:
        """
        Handle one confirmed label from the delayed-label buffer.

        1. Update performance drift monitor (ADWIN/PH)
        2. Append to confirmed buffer for candidate training
        3. Update rolling metrics
        4. Trigger adaptation check if severity threshold met
        5. Update post-deployment monitor if active
        """
        # Store for candidate training
        self._confirmed_buffer.append(pending)

        # Keep only the recent window
        cutoff = pending.tx_datetime - timedelta(days=self.candidate_window_days * 2)
        self._confirmed_buffer = [
            p for p in self._confirmed_buffer if p.tx_datetime >= cutoff
        ]

        # Update rolling metrics
        self._rolling.update(pending.y_true, pending.y_pred, pending.y_prob)

        if not self.flags["use_drift_detection"]:
            return

        # Update performance monitor (requires confirmed label)
        perf_result = self.perf_monitor.update(pending.y_true, pending.y_pred)

        # Severity assessment (data + perf)
        assessment = self.severity_est.assess(
            data_result = self._last_data_result,
            perf_result = perf_result,
        )
        self._last_assessment = assessment

        if not self.flags["use_severity"]:
            # Skip severity check — adapt on any ADWIN signal (ablation A2)
            should_adapt = perf_result.adwin_detected
        else:
            should_adapt = assessment.level >= self.min_severity_to_adapt

        # Post-deployment monitoring
        if self._post_deploy_active:
            self._monitor_post_deployment(pending, assessment)

        # Trigger adaptation
        if should_adapt:
            self._run_adaptation_cycle(pending.tx_datetime, assessment)

    def monitor_data_drift(
        self,
        current_df: pd.DataFrame,
        timestamp:  pd.Timestamp,
    ) -> Optional[DataDriftResult]:
        """
        Run label-free data drift monitoring on a batch.
        Called periodically during streaming (e.g., daily batch).
        """
        if not self.flags["use_drift_detection"]:
            return None

        result = self.data_monitor.monitor(
            current_df = current_df,
            timestamp  = timestamp,
            run_mmd    = False,  # Enable for advanced experiments
        )
        self._last_data_result = result
        return result

    # ------------------------------------------------------------------
    # Adaptation cycle
    # ------------------------------------------------------------------

    def _run_adaptation_cycle(
        self,
        timestamp:  pd.Timestamp,
        assessment: SeverityAssessment,
    ) -> None:
        """
        Full adaptation cycle:
        1. Split confirmed buffer into training / validation windows
        2. Train candidate model
        3. Run validation gate
        4. Promote or reject
        """
        log.info(
            f"[Controller] Adaptation triggered | "
            f"Severity={assessment.level} | "
            f"Score={assessment.score:.3f}"
        )

        # --- Build training and validation windows ---
        if len(self._confirmed_buffer) < 100:
            log.warning("[Controller] Insufficient confirmed labels for adaptation. Skipping.")
            return

        # Convert buffer to DataFrame
        data = {
            "datetime": [p.tx_datetime for p in self._confirmed_buffer],
            "y_true":   [p.y_true      for p in self._confirmed_buffer],
        }
        feat_data = pd.DataFrame([p.features for p in self._confirmed_buffer])
        feat_data["datetime"] = [p.tx_datetime for p in self._confirmed_buffer]
        feat_data["y_true"]   = [p.y_true      for p in self._confirmed_buffer]

        # Split: training window | validation window
        val_cutoff = feat_data["datetime"].max() - timedelta(days=self.validation_window_days)
        train_mask = feat_data["datetime"] < val_cutoff
        val_mask   = feat_data["datetime"] >= val_cutoff

        X_train = feat_data[train_mask].drop(columns=["datetime", "y_true"])
        y_train = feat_data[train_mask]["y_true"]
        X_val   = feat_data[val_mask].drop(columns=["datetime", "y_true"])
        y_val   = feat_data[val_mask]["y_true"]

        if len(X_train) < 50 or len(X_val) < 10:
            log.warning("[Controller] Windows too small. Skipping adaptation.")
            return

        # --- Train candidate ---
        try:
            cand_model = self.candidate_trainer.train(X_train, y_train)
        except Exception as e:
            log.error(f"[Controller] Candidate training failed: {e}")
            return

        prod_version = self.registry.production_version

        # Register candidate
        cand_version = self.registry.register(
            model        = cand_model,
            name         = "adaptive_candidate",
            train_start  = str(feat_data["datetime"].min()),
            train_end    = str(val_cutoff),
            metrics      = {},
            hyperparams  = {},
            parent_version = prod_version,
            status       = "candidate",
        )

        # --- Validation gate ---
        if self.flags["use_validation_gate"]:
            prod_model, _ = self.registry.get_production_model()

            val_result = self.validation_gate.evaluate(
                candidate_model    = cand_model,
                production_model   = prod_model,
                X_val              = X_val,
                y_val              = y_val,
                candidate_version  = cand_version,
                production_version = prod_version,
            )

            if val_result.gate_passed:
                self.registry.promote(cand_version)
                self.adaptation_count += 1
                self._post_deploy_active = True
                self._post_deploy_buffer = []
                self.perf_monitor.reset_detectors()

                event = AdaptationEvent(
                    timestamp        = str(pd.Timestamp.now()),
                    severity_level   = str(assessment.level),
                    severity_score   = assessment.score,
                    action           = "adapted",
                    production_v     = cand_version,
                    candidate_v      = cand_version,
                    validation_result = "PASSED",
                )
            else:
                # REJECTION — candidate never reaches production
                self.registry.reject(
                    cand_version,
                    reason=val_result.rejection_reason,
                )
                self.rejection_count += 1

                event = AdaptationEvent(
                    timestamp        = str(pd.Timestamp.now()),
                    severity_level   = str(assessment.level),
                    severity_score   = assessment.score,
                    action           = "rejected",
                    production_v     = prod_version,
                    candidate_v      = cand_version,
                    validation_result = "REJECTED",
                )
        else:
            # Ablation A4: No validation gate — promote directly
            self.registry.promote(cand_version)
            self.adaptation_count += 1
            self.perf_monitor.reset_detectors()
            event = AdaptationEvent(
                timestamp      = str(pd.Timestamp.now()),
                severity_level = str(assessment.level),
                severity_score = assessment.score,
                action         = "adapted_no_gate",
                production_v   = cand_version,
                candidate_v    = cand_version,
            )

        self.adaptation_events.append(event)

    # ------------------------------------------------------------------
    # Post-deployment monitoring
    # ------------------------------------------------------------------

    def _monitor_post_deployment(
        self,
        pending:    PendingLabel,
        assessment: SeverityAssessment,
    ) -> None:
        """
        Monitor model performance after a successful promotion.
        If degradation detected, trigger rollback.
        """
        if not self.flags["use_rollback"]:
            return

        self._post_deploy_buffer.append({
            "y_true": pending.y_true,
            "y_pred": pending.y_pred,
        })

        # Check after accumulating enough post-deployment observations
        if len(self._post_deploy_buffer) < 200:
            return

        y_true_post = np.array([x["y_true"] for x in self._post_deploy_buffer])
        y_pred_post = np.array([x["y_pred"] for x in self._post_deploy_buffer])

        post_metrics = evaluate_predictions(
            y_true_post,
            y_pred_post,
            y_pred_post.astype(float),
        )

        recall = post_metrics.get("recall", 1.0)

        # Rollback if post-deployment recall drops critically
        recall_floor = self.cfg["adaptation"]["gate"]["recall_floor"]
        if recall < recall_floor * 0.85:  # 15% below the promotion floor
            log.warning(
                f"[Controller] Post-deployment degradation detected! "
                f"Recall={recall:.4f} < threshold. Triggering ROLLBACK."
            )
            try:
                restored_v = self.registry.rollback(
                    reason=f"Post-deployment recall degradation: {recall:.4f}"
                )
                self._post_deploy_active = False
                self._post_deploy_buffer = []
                self.perf_monitor.reset_detectors()

                event = AdaptationEvent(
                    timestamp      = str(pd.Timestamp.now()),
                    severity_level = "CRITICAL",
                    severity_score = 1.0,
                    action         = "rollback",
                    production_v   = restored_v,
                )
                self.adaptation_events.append(event)
            except RuntimeError as e:
                log.error(f"[Controller] Rollback failed: {e}")

    # ------------------------------------------------------------------
    # Status & reporting
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current system status for API/dashboard."""
        assessment = self._last_assessment
        return {
            "production_version":    self.registry.production_version,
            "severity_level":        str(assessment.level) if assessment else "NONE",
            "severity_score":        assessment.score if assessment else 0.0,
            "adwin_signal":          assessment.adwin_signal if assessment else False,
            "max_psi":               assessment.max_psi if assessment else 0.0,
            "error_rate":            self.perf_monitor.current_error_rate,
            "adaptation_count":      self.adaptation_count,
            "rejection_count":       self.rejection_count,
            "rollback_count":        self.registry.rollback_count,
            "pending_labels":        self.label_buffer.peek_buffer_size(),
            "confirmed_labels":      len(self._confirmed_buffer),
        }

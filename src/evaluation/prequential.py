"""
AdaptGuard AI — Prequential Evaluation Engine
Implements the test-then-train streaming protocol for chronological evaluation.

Protocol (per transaction):
1. Generate prediction (before label is known)
2. Store prediction + features in delayed-label buffer
3. Advance simulation clock
4. Release confirmed labels from buffer
5. Update performance monitor (ADWIN/PH)
6. Update online/adaptive model
7. Monitor drift (data + performance channels)
8. Trigger adaptation if severity threshold met
9. Record metrics

NO random shuffling. NO global train_test_split.
All models evaluated under the same temporal framework.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from datetime import timedelta

from src.utils.logger import get_logger
from src.evaluation.metrics import evaluate_predictions, RollingMetrics

log = get_logger("evaluation.prequential")


@dataclass
class StreamResult:
    """Complete results from one prequential evaluation run."""
    model_name:         str
    n_transactions:     int
    n_fraud_detected:   int
    n_fraud_total:      int
    adaptation_events:  int
    rejection_count:    int
    rollback_count:     int
    final_metrics:      dict = field(default_factory=dict)
    rolling_history:    list = field(default_factory=list)   # PR-AUC over time
    drift_events:       list = field(default_factory=list)
    label_delay_days:   int = 3


class PrequentialEvaluator:
    """
    Runs the prequential (test-then-train) streaming evaluation.

    Supports all four model configurations:
    - Static XGBoost (no updates)
    - Periodic Retraining (update every N days)
    - Always-Online (update on every confirmed label)
    - AdaptGuard AI (selective adaptation based on drift severity)
    """

    def __init__(
        self,
        model:              Any,
        model_name:         str,
        delay_days:         int = 3,
        decision_threshold: float = 0.5,
        rolling_window:     int = 1000,
    ):
        """
        Args:
            model:              A model with .predict_proba() method.
            model_name:         Identifier string for results.
            delay_days:         Label delay (0=oracle, 1, 3, 7).
            decision_threshold: Fraud decision threshold.
            rolling_window:     Window size for rolling metric tracking.
        """
        self.model              = model
        self.model_name         = model_name
        self.delay_days         = delay_days
        self.decision_threshold = decision_threshold
        self.rolling_metrics    = RollingMetrics(window_size=rolling_window)

        self.n_processed        = 0
        self.adaptation_events  = 0
        self.rejection_count    = 0
        self.rollback_count     = 0
        self.drift_events: list = []

    def predict_one(self, X_row: pd.DataFrame) -> tuple[int, float]:
        """
        Generate a single-row prediction.

        Returns:
            (binary_prediction, fraud_probability)
        """
        proba = self.model.predict_proba(X_row)
        if hasattr(proba, '__len__') and len(proba.shape) > 1:
            proba = proba[:, 1]
        prob  = float(proba[0]) if hasattr(proba, '__len__') else float(proba)
        pred  = int(prob >= self.decision_threshold)
        return pred, prob

    def run(
        self,
        df:                  pd.DataFrame,
        feature_cols:        list[str],
        on_label_released:   Optional[Callable] = None,
        on_drift_detected:   Optional[Callable] = None,
        verbose_every:       int = 10000,
    ) -> StreamResult:
        """
        Run prequential evaluation over the full stream.

        Args:
            df:                 Full transaction DataFrame (chronologically sorted).
            feature_cols:       Feature column names.
            on_label_released:  Callback when a label is released.
                                Signature: (pending_label) → None
            on_drift_detected:  Callback when drift is detected.
                                Signature: (severity_assessment) → None
            verbose_every:      Log progress every N transactions.

        Returns:
            StreamResult with all metrics and history.
        """
        from src.adaptation.delayed_labels import DelayedLabelBuffer

        label_buffer = DelayedLabelBuffer(delay_days=self.delay_days)

        all_y_true:  list = []
        all_y_pred:  list = []
        all_y_proba: list = []

        log.info(
            f"[Prequential] Starting evaluation: model={self.model_name} | "
            f"delay={self.delay_days}d | "
            f"n_tx={len(df):,}"
        )

        for i, (idx, row) in enumerate(df.iterrows()):
            X_row = pd.DataFrame([row[feature_cols]])
            y_true = int(row["TX_FRAUD"])
            tx_dt  = row["TX_DATETIME"]

            # Step 1: PREDICT (before label is known)
            y_pred, y_prob = self.predict_one(X_row)

            # Step 2: Store in delayed-label buffer
            label_buffer.store(
                transaction_id = int(row.get("TRANSACTION_ID", i)),
                tx_datetime    = tx_dt,
                y_true         = y_true,
                y_pred         = y_pred,
                y_prob         = y_prob,
                features       = row[feature_cols],
            )

            # Step 3: Release labels whose delay has elapsed
            confirmed = label_buffer.release(current_time=tx_dt)
            for pending in confirmed:
                all_y_true.append(pending.y_true)
                all_y_pred.append(pending.y_pred)
                all_y_proba.append(pending.y_prob)
                self.rolling_metrics.update(
                    pending.y_true, pending.y_pred, pending.y_prob, tx_dt
                )
                if on_label_released:
                    on_label_released(pending)

            self.n_processed += 1

            if verbose_every > 0 and (i + 1) % verbose_every == 0:
                n_conf = len(all_y_true)
                recent_prauc = (
                    self.rolling_metrics.history[-1]["pr_auc"]
                    if self.rolling_metrics.history else 0.0
                )
                log.info(
                    f"  [{self.model_name}] Processed {i+1:,} tx | "
                    f"Confirmed labels: {n_conf:,} | "
                    f"Rolling PR-AUC: {recent_prauc:.4f}"
                )

        # Final metrics over all confirmed labels
        if all_y_true:
            final_metrics = evaluate_predictions(
                np.array(all_y_true),
                np.array(all_y_pred),
                np.array(all_y_proba),
                threshold=self.decision_threshold,
            )
        else:
            final_metrics = {}

        log.info(
            f"[Prequential] Complete: {self.model_name} | "
            f"PR-AUC={final_metrics.get('pr_auc', 'N/A'):.4f} | "
            f"Recall={final_metrics.get('recall', 'N/A'):.4f} | "
            f"Adaptations={self.adaptation_events} | "
            f"Rejections={self.rejection_count} | "
            f"Rollbacks={self.rollback_count}"
            if final_metrics else f"[Prequential] Complete: {self.model_name}"
        )

        return StreamResult(
            model_name        = self.model_name,
            n_transactions    = self.n_processed,
            n_fraud_detected  = final_metrics.get("tp", 0),
            n_fraud_total     = final_metrics.get("n_fraud", 0),
            adaptation_events = self.adaptation_events,
            rejection_count   = self.rejection_count,
            rollback_count    = self.rollback_count,
            final_metrics     = final_metrics,
            rolling_history   = self.rolling_metrics.history.copy(),
            drift_events      = self.drift_events.copy(),
            label_delay_days  = self.delay_days,
        )

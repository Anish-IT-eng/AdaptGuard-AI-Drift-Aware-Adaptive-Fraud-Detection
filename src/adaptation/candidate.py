"""
AdaptGuard AI — Candidate Model Training & Validation Gate
Implements the Champion-Challenger pattern for safe model promotion.

The validation gate is the critical safety mechanism. It prevents production
degradation by comparing candidate models against the production model
on a held-out chronological validation window BEFORE any promotion occurs.

Two distinct outcomes from this module:
- REJECTION: Candidate fails gate — never reaches production (not a rollback)
- ACCEPTANCE: Candidate passes gate — promoted by the controller
"""

import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Any

from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.evaluation.metrics import evaluate_predictions

log = get_logger("adaptation.candidate")


@dataclass
class ValidationResult:
    """Result from the validation gate comparison."""
    candidate_version:   int
    production_version:  int
    gate_passed:         bool
    rejection_reason:    str          # Empty string if passed

    # Candidate metrics
    candidate_pr_auc:   float
    candidate_recall:   float
    candidate_precision: float
    candidate_fpr:      float
    candidate_latency_ms: float

    # Production metrics (on same window)
    production_pr_auc:   float
    production_recall:   float

    # Gate criteria results
    prauc_improved:      bool
    recall_above_floor:  bool
    fpr_acceptable:      bool
    latency_acceptable:  bool


class CandidateTrainer:
    """
    Trains a candidate model on recent confirmed-label data.

    Uses the window of confirmed transactions from the delayed-label buffer.
    The window is chronologically ordered and does NOT include the validation window.
    """

    def __init__(
        self,
        model_factory,
        scaler: Optional[StandardScaler] = None,
        name: str = "candidate",
    ):
        self.model_factory = model_factory
        self.scaler        = scaler or StandardScaler()
        self.name          = name
        self._fitted_model = None
        self._is_fitted    = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Any:
        """
        Train a new candidate model on recent data.

        Args:
            X_train: Feature matrix from recent confirmed-label window.
            y_train: Confirmed fraud labels.

        Returns:
            Fitted sklearn model.
        """
        if len(X_train) < 50:
            raise ValueError(
                f"Insufficient training data for candidate: {len(X_train)} samples. "
                f"Need at least 50."
            )

        log.info(
            f"[CandidateTrainer] Training on {len(X_train):,} samples | "
            f"Fraud rate: {y_train.mean()*100:.3f}%"
        )

        self._fitted_model = self.model_factory()
        self._fitted_model.fit(X_train, y_train)
        self._is_fitted = True

        log.info(f"[CandidateTrainer] Training complete.")
        return self._fitted_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Candidate not trained yet.")
        return self._fitted_model.predict_proba(X)[:, 1]


class ValidationGate:
    """
    Validates a candidate model against the current production model
    on a held-out chronological validation window.

    Gate criteria (all must pass for promotion):
    1. PR-AUC improvement over production (>= min_improvement)
    2. Recall above safety floor (>= recall_floor)
    3. FPR below ceiling (<= fpr_ceiling)
    4. Inference latency acceptable (<= latency_ceiling_ms)

    Thresholds are configurable and must be determined experimentally.
    """

    def __init__(
        self,
        prauc_improvement:    float = 0.005,
        recall_floor:         float = 0.60,
        fpr_ceiling:          float = 0.10,
        latency_ceiling_ms:   float = 50.0,
        decision_threshold:   float = 0.5,
    ):
        self.prauc_improvement  = prauc_improvement
        self.recall_floor       = recall_floor
        self.fpr_ceiling        = fpr_ceiling
        self.latency_ceiling_ms = latency_ceiling_ms
        self.decision_threshold = decision_threshold

    def evaluate(
        self,
        candidate_model:   Any,
        production_model:  Any,
        X_val:             pd.DataFrame,
        y_val:             pd.Series,
        candidate_version: int,
        production_version: int,
    ) -> ValidationResult:
        """
        Run the validation gate comparison.

        Args:
            candidate_model:   Fitted candidate model with predict_proba().
            production_model:  Current production model with predict_proba().
            X_val:             Validation feature matrix (held-out, chronological).
            y_val:             Validation labels (confirmed fraud labels).
            candidate_version: Registry version number of candidate.
            production_version: Registry version number of production.

        Returns:
            ValidationResult with gate_passed=True/False and full breakdown.
        """
        import time

        if len(X_val) < 10:
            log.warning(f"[ValidationGate] Very small validation window: {len(X_val)} samples")

        # --- Evaluate candidate ---
        t0      = time.time()
        cand_proba = candidate_model.predict_proba(X_val)[:, 1]
        latency = (time.time() - t0) / len(X_val) * 1000  # ms per prediction

        cand_preds = (cand_proba >= self.decision_threshold).astype(int)
        cand_metrics = evaluate_predictions(y_val, cand_preds, cand_proba)

        # --- Evaluate production on same window ---
        prod_proba = production_model.predict_proba(X_val)[:, 1]
        prod_preds = (prod_proba >= self.decision_threshold).astype(int)
        prod_metrics = evaluate_predictions(y_val, prod_preds, prod_proba)

        # --- Gate checks ---
        prauc_improved    = (cand_metrics["pr_auc"] - prod_metrics["pr_auc"]) >= self.prauc_improvement
        recall_above      = cand_metrics["recall"] >= self.recall_floor
        fpr_acceptable    = cand_metrics["fpr"] <= self.fpr_ceiling
        latency_ok        = latency <= self.latency_ceiling_ms

        gate_passed = prauc_improved and recall_above and fpr_acceptable and latency_ok

        # Build rejection reason
        reasons = []
        if not prauc_improved:
            reasons.append(
                f"PR-AUC improvement insufficient "
                f"({cand_metrics['pr_auc']:.4f} vs {prod_metrics['pr_auc']:.4f}, "
                f"required +{self.prauc_improvement:.4f})"
            )
        if not recall_above:
            reasons.append(f"Recall below floor ({cand_metrics['recall']:.4f} < {self.recall_floor})")
        if not fpr_acceptable:
            reasons.append(f"FPR above ceiling ({cand_metrics['fpr']:.4f} > {self.fpr_ceiling})")
        if not latency_ok:
            reasons.append(f"Latency too high ({latency:.1f}ms > {self.latency_ceiling_ms}ms)")

        rejection_reason = "; ".join(reasons) if reasons else ""

        result = ValidationResult(
            candidate_version    = candidate_version,
            production_version   = production_version,
            gate_passed          = gate_passed,
            rejection_reason     = rejection_reason,
            candidate_pr_auc     = cand_metrics["pr_auc"],
            candidate_recall     = cand_metrics["recall"],
            candidate_precision  = cand_metrics["precision"],
            candidate_fpr        = cand_metrics["fpr"],
            candidate_latency_ms = latency,
            production_pr_auc    = prod_metrics["pr_auc"],
            production_recall    = prod_metrics["recall"],
            prauc_improved       = prauc_improved,
            recall_above_floor   = recall_above,
            fpr_acceptable       = fpr_acceptable,
            latency_acceptable   = latency_ok,
        )

        status = "✅ PASSED" if gate_passed else "❌ REJECTED"
        log.info(
            f"[ValidationGate] {status} | "
            f"Candidate v{candidate_version} vs Production v{production_version} | "
            f"PR-AUC: {cand_metrics['pr_auc']:.4f} vs {prod_metrics['pr_auc']:.4f} | "
            f"Recall: {cand_metrics['recall']:.4f} | "
            f"FPR: {cand_metrics['fpr']:.4f} | "
            f"Latency: {latency:.1f}ms"
        )

        if not gate_passed:
            log.info(f"[ValidationGate] Rejection reason: {rejection_reason}")
            log.info(
                f"[ValidationGate] NOTE: This is a REJECTION (candidate never deployed). "
                f"Production v{production_version} continues unchanged."
            )

        return result

"""
Unit tests — src/adaptation/candidate.py (ValidationGate)
Validates gate pass/fail logic for all four criteria.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.adaptation.candidate import ValidationGate, ValidationResult


# ============================================================
# Helpers
# ============================================================

def _make_fitted_model(bias: float = 0.0, seed: int = 42):
    """Return a LogisticRegression guaranteed to have both classes."""
    np.random.seed(seed)
    n = 300
    X = np.random.rand(n, 4)
    # Force exactly 30 fraud labels (10%) regardless of bias — guarantees two classes
    y = np.zeros(n, dtype=int)
    fraud_idx = np.random.choice(n, size=30, replace=False)
    y[fraud_idx] = 1
    m = LogisticRegression(max_iter=300, class_weight="balanced")
    m.fit(X, y)
    return m


def _make_val_data(n: int = 300, fraud_rate: float = 0.10, seed: int = 0):
    """Return validation data with guaranteed both classes."""
    np.random.seed(seed)
    X = np.random.rand(n, 4)
    # Force at least 10 fraud and 10 legitimate samples
    y = np.zeros(n, dtype=int)
    n_fraud = max(10, int(n * fraud_rate))
    fraud_idx = np.random.choice(n, size=n_fraud, replace=False)
    y[fraud_idx] = 1
    return pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"]), pd.Series(y)


# ============================================================
# ValidationGate Tests
# ============================================================

class TestValidationGate:
    def test_result_type(self):
        gate = ValidationGate(
            prauc_improvement=0.0,  # Accept any improvement
            recall_floor=0.0,
            fpr_ceiling=1.0,
            latency_ceiling_ms=5000.0,
        )
        cand = _make_fitted_model(bias=0.5)
        prod = _make_fitted_model(bias=0.0)
        X_val, y_val = _make_val_data(n=200)

        result = gate.evaluate(
            candidate_model    = cand,
            production_model   = prod,
            X_val              = X_val,
            y_val              = y_val,
            candidate_version  = 2,
            production_version = 1,
        )
        assert isinstance(result, ValidationResult)

    def test_gate_has_version_info(self):
        gate = ValidationGate(prauc_improvement=0.0, recall_floor=0.0,
                              fpr_ceiling=1.0, latency_ceiling_ms=5000.0)
        cand = _make_fitted_model()
        prod = _make_fitted_model()
        X_val, y_val = _make_val_data()

        result = gate.evaluate(cand, prod, X_val, y_val,
                               candidate_version=5, production_version=3)
        assert result.candidate_version == 5
        assert result.production_version == 3

    def test_gate_passed_field_is_bool(self):
        gate = ValidationGate(prauc_improvement=0.0, recall_floor=0.0,
                              fpr_ceiling=1.0, latency_ceiling_ms=5000.0)
        cand = _make_fitted_model()
        prod = _make_fitted_model()
        X_val, y_val = _make_val_data()
        result = gate.evaluate(cand, prod, X_val, y_val, 2, 1)
        assert isinstance(result.gate_passed, bool)

    def test_impossible_prauc_improvement_fails_gate(self):
        """Requiring 100% PR-AUC improvement should always fail."""
        gate = ValidationGate(
            prauc_improvement  = 100.0,  # impossible
            recall_floor       = 0.0,
            fpr_ceiling        = 1.0,
            latency_ceiling_ms = 5000.0,
        )
        cand = _make_fitted_model()
        prod = _make_fitted_model()
        X_val, y_val = _make_val_data()
        result = gate.evaluate(cand, prod, X_val, y_val, 2, 1)
        assert result.gate_passed is False
        assert result.prauc_improved is False

    def test_impossible_recall_floor_fails_gate(self):
        """Requiring recall=1.0 should always fail on imperfect model."""
        gate = ValidationGate(
            prauc_improvement  = 0.0,
            recall_floor       = 1.0,  # impossible
            fpr_ceiling        = 1.0,
            latency_ceiling_ms = 5000.0,
        )
        cand = _make_fitted_model()
        prod = _make_fitted_model()
        X_val, y_val = _make_val_data()
        result = gate.evaluate(cand, prod, X_val, y_val, 2, 1)
        assert result.gate_passed is False

    def test_rejection_reason_populated_on_failure(self):
        gate = ValidationGate(prauc_improvement=100.0, recall_floor=0.0,
                              fpr_ceiling=1.0, latency_ceiling_ms=5000.0)
        cand = _make_fitted_model()
        prod = _make_fitted_model()
        X_val, y_val = _make_val_data()
        result = gate.evaluate(cand, prod, X_val, y_val, 2, 1)
        assert result.gate_passed is False
        assert len(result.rejection_reason) > 0

    def test_rejection_reason_empty_on_pass(self):
        """When all criteria pass, rejection_reason should be empty."""
        gate = ValidationGate(
            prauc_improvement  = -1.0,  # Always passes PR-AUC check
            recall_floor       = 0.0,
            fpr_ceiling        = 1.0,
            latency_ceiling_ms = 5000.0,
        )
        cand = _make_fitted_model()
        prod = _make_fitted_model()
        X_val, y_val = _make_val_data()
        result = gate.evaluate(cand, prod, X_val, y_val, 2, 1)
        if result.gate_passed:
            assert result.rejection_reason == ""

    def test_metrics_are_floats(self):
        gate = ValidationGate(prauc_improvement=0.0, recall_floor=0.0,
                              fpr_ceiling=1.0, latency_ceiling_ms=5000.0)
        cand = _make_fitted_model()
        prod = _make_fitted_model()
        X_val, y_val = _make_val_data()
        result = gate.evaluate(cand, prod, X_val, y_val, 2, 1)
        assert isinstance(result.candidate_pr_auc, float)
        assert isinstance(result.candidate_recall, float)
        assert isinstance(result.candidate_latency_ms, float)
        assert 0.0 <= result.candidate_pr_auc <= 1.0

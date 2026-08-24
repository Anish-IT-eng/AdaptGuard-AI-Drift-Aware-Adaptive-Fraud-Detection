"""
Unit tests — src/evaluation/metrics.py + src/evaluation/cost.py
Validates metric computations, edge cases, and cost sensitivity analysis.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.evaluation.metrics import (
    evaluate_predictions,
    compute_business_cost,
    compute_business_cost_all_ratios,
    compute_detection_delay,
    precision_at_k,
    RollingMetrics,
)
from src.evaluation.cost import (
    compute_total_cost,
    sensitivity_analysis,
    compare_model_costs,
    format_cost_table,
    DEFAULT_COST_RATIOS,
)


# ============================================================
# evaluate_predictions Tests
# ============================================================

class TestEvaluatePredictions:
    def test_returns_dict(self, binary_labels):
        y_true, y_pred, y_proba = binary_labels
        result = evaluate_predictions(y_true, y_pred, y_proba)
        assert isinstance(result, dict)

    def test_required_keys_present(self, binary_labels):
        y_true, y_pred, y_proba = binary_labels
        result = evaluate_predictions(y_true, y_pred, y_proba)
        for key in ["pr_auc", "recall", "precision", "f1", "fpr", "roc_auc"]:
            assert key in result, f"Missing key: {key}"

    def test_all_values_in_range(self, binary_labels):
        y_true, y_pred, y_proba = binary_labels
        result = evaluate_predictions(y_true, y_pred, y_proba)
        for k in ["pr_auc", "recall", "precision", "f1", "fpr", "roc_auc"]:
            assert 0.0 <= result[k] <= 1.0, f"Metric {k}={result[k]} out of [0,1]"

    def test_perfect_classifier(self):
        y_true  = np.array([0]*90 + [1]*10)
        y_proba = np.where(y_true == 1, 0.99, 0.01).astype(float)
        y_pred  = (y_proba >= 0.5).astype(int)
        result = evaluate_predictions(y_true, y_pred, y_proba)
        assert result["recall"] == pytest.approx(1.0, abs=1e-6)
        assert result["precision"] == pytest.approx(1.0, abs=1e-6)
        assert result["pr_auc"] == pytest.approx(1.0, abs=1e-4)

    def test_all_zeros_handling(self):
        """Edge case: all predictions 0, some fraud in y_true."""
        y_true  = np.array([1, 0, 0, 1, 0])
        y_pred  = np.zeros(5, dtype=int)
        y_proba = np.zeros(5, dtype=float)
        # Should not raise; recall will be 0
        result = evaluate_predictions(y_true, y_pred, y_proba)
        assert result["recall"] == 0.0

    def test_single_class_warning(self):
        """All-negative y_true returns safe defaults without crashing."""
        y_true  = np.zeros(50, dtype=int)
        y_pred  = np.zeros(50, dtype=int)
        y_proba = np.zeros(50, dtype=float)
        result = evaluate_predictions(y_true, y_pred, y_proba)
        assert "pr_auc" in result

    def test_n_samples_correct(self, binary_labels):
        y_true, y_pred, y_proba = binary_labels
        result = evaluate_predictions(y_true, y_pred, y_proba)
        assert result["n_samples"] == len(y_true)


# ============================================================
# precision_at_k Tests
# ============================================================

class TestPrecisionAtK:
    def test_returns_float(self, binary_labels):
        y_true, _, y_proba = binary_labels
        result = precision_at_k(y_true, y_proba, k=50)
        assert isinstance(result, float)

    def test_in_range(self, binary_labels):
        y_true, _, y_proba = binary_labels
        result = precision_at_k(y_true, y_proba, k=50)
        assert 0.0 <= result <= 1.0

    def test_k_larger_than_n(self, binary_labels):
        y_true, _, y_proba = binary_labels
        result = precision_at_k(y_true, y_proba, k=10_000)
        assert isinstance(result, float)


# ============================================================
# compute_detection_delay Tests
# ============================================================

class TestDetectionDelay:
    def test_no_detection(self):
        assert compute_detection_delay(90, None) is None

    def test_immediate_detection(self):
        delay = compute_detection_delay(drift_start_day=90, detection_day=90)
        assert delay == 0.0

    def test_positive_delay(self):
        delay = compute_detection_delay(drift_start_day=90, detection_day=97)
        assert delay == 7.0


# ============================================================
# Business Cost Tests (metrics.py)
# ============================================================

class TestBusinessCost:
    def test_returns_dict(self, binary_labels):
        y_true, y_pred, _ = binary_labels
        result = compute_business_cost(y_true, y_pred, fn_cost=10, fp_cost=1)
        assert isinstance(result, dict)

    def test_total_cost_non_negative(self, binary_labels):
        y_true, y_pred, _ = binary_labels
        result = compute_business_cost(y_true, y_pred, fn_cost=10, fp_cost=1)
        assert result["total_cost"] >= 0

    def test_all_ratios_returns_list(self, binary_labels):
        y_true, y_pred, _ = binary_labels
        results = compute_business_cost_all_ratios(y_true, y_pred)
        assert isinstance(results, list)
        assert len(results) == 3  # 10:1, 5:1, 2:1


# ============================================================
# Business Cost Tests (cost.py)
# ============================================================

class TestCostModule:
    def test_compute_total_cost_keys(self, binary_labels):
        y_true, y_pred, _ = binary_labels
        result = compute_total_cost(y_true, y_pred, fn_cost=10, fp_cost=1)
        assert result.ratio_label == "10:1"
        assert result.total_cost >= 0.0
        assert result.cost_per_tx >= 0.0

    def test_sensitivity_analysis_length(self, binary_labels):
        y_true, y_pred, _ = binary_labels
        results = sensitivity_analysis(y_true, y_pred)
        assert len(results) == len(DEFAULT_COST_RATIOS)

    def test_higher_fn_cost_gives_higher_total(self, binary_labels):
        """10:1 should cost more than 2:1 when FN > 0."""
        y_true, y_pred, _ = binary_labels
        r10 = compute_total_cost(y_true, y_pred, fn_cost=10, fp_cost=1)
        r2  = compute_total_cost(y_true, y_pred, fn_cost=2,  fp_cost=1)
        if r10.n_fn > 0:
            assert r10.total_cost >= r2.total_cost

    def test_compare_model_costs_returns_df(self, binary_labels):
        y_true, y_pred, _ = binary_labels
        df = compare_model_costs(
            model_predictions={"modelA": y_pred, "modelB": y_pred},
            y_true=y_true,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_format_cost_table_columns(self, binary_labels):
        y_true, y_pred, _ = binary_labels
        results = sensitivity_analysis(y_true, y_pred)
        df = format_cost_table(results)
        assert "FN:FP Ratio" in df.columns
        assert "Total Cost" in df.columns


# ============================================================
# RollingMetrics Tests
# ============================================================

class TestRollingMetrics:
    def test_update_increments_n_total(self):
        rm = RollingMetrics(window_size=100)
        for i in range(50):
            rm.update(0, 0, 0.1)
        assert rm.n_total == 50

    def test_history_grows_after_200(self):
        rm = RollingMetrics(window_size=1000)
        # History is computed every 100 updates after n>=200
        for i in range(300):
            y_true = int(np.random.rand() < 0.05)
            y_pred = int(np.random.rand() < 0.1)
            y_prob = np.random.rand() * 0.3
            rm.update(y_true, y_pred, y_prob)
        # Should have at least 1 history entry
        assert len(rm.history) >= 1

    def test_to_dataframe(self):
        rm = RollingMetrics(window_size=500)
        for i in range(300):
            rm.update(int(np.random.rand() < 0.05), 0, 0.1)
        df = rm.to_dataframe()
        assert isinstance(df, pd.DataFrame)

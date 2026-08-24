"""
Unit tests — src/models/registry.py
Validates model registration, promotion, rejection, and rollback lifecycle.
"""

import pytest
import numpy as np
import sys
import tempfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.registry import ModelRegistry, ModelRecord


# ============================================================
# Helpers
# ============================================================

def _make_model():
    """Return a tiny fitted LogisticRegression for registry tests."""
    X = np.random.rand(100, 3)
    y = (np.random.rand(100) > 0.5).astype(int)
    m = LogisticRegression(max_iter=200)
    m.fit(X, y)
    return m


# ============================================================
# Registration Tests
# ============================================================

class TestModelRegistryRegistration:
    def test_register_returns_version(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        model = _make_model()
        v = reg.register(
            model        = model,
            name         = "test_model",
            train_start  = "2023-01-01",
            train_end    = "2023-02-01",
            metrics      = {"pr_auc": 0.75},
            hyperparams  = {"n_estimators": 100},
        )
        assert v == 1

    def test_version_increments(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v1 = reg.register(m, "m1", "2023-01-01", "2023-02-01", {}, {})
        v2 = reg.register(m, "m2", "2023-02-01", "2023-03-01", {}, {})
        assert v2 == v1 + 1

    def test_model_file_saved(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v = reg.register(m, "test", "2023-01-01", "2023-02-01", {}, {})
        model_file = tmp_path / f"model_v{v}_test.joblib"
        assert model_file.exists()

    def test_registry_index_saved(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        reg.register(m, "test", "2023-01-01", "2023-02-01", {}, {})
        assert (tmp_path / "registry_index.json").exists()


# ============================================================
# Promotion Tests
# ============================================================

class TestModelRegistryPromotion:
    def test_promote_sets_production(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v = reg.register(m, "cand", "2023-01-01", "2023-02-01", {}, {}, status="candidate")
        reg.promote(v)
        assert reg.production_version == v

    def test_old_production_archived(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v1 = reg.register(m, "v1", "2023-01-01", "2023-02-01", {}, {}, status="production")
        v2 = reg.register(m, "v2", "2023-02-01", "2023-03-01", {}, {}, status="candidate")
        reg.promote(v2)
        assert reg.get_record(v1).status == "archived"
        assert reg.get_record(v2).status == "production"

    def test_promote_unknown_version_raises(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        with pytest.raises(KeyError):
            reg.promote(999)


# ============================================================
# Rejection Tests
# ============================================================

class TestModelRegistryRejection:
    def test_reject_increments_counter(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v = reg.register(m, "cand", "2023-01-01", "2023-02-01", {}, {}, status="candidate")
        reg.reject(v, reason="pr_auc too low")
        assert reg.rejected_count == 1

    def test_reject_does_not_change_production(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v1 = reg.register(m, "prod", "2023-01-01", "2023-02-01", {}, {}, status="production")
        v2 = reg.register(m, "cand", "2023-02-01", "2023-03-01", {}, {}, status="candidate")
        reg.reject(v2, reason="failed gate")
        assert reg.production_version == v1


# ============================================================
# Rollback Tests
# ============================================================

class TestModelRegistryRollback:
    def test_rollback_restores_previous(self, tmp_path):
        """v1 must go through promote() so promoted_at is set and rollback can find it."""
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        # Register v1 as candidate then promote — sets promoted_at
        v1 = reg.register(m, "v1", "2023-01-01", "2023-02-01", {}, {}, status="candidate")
        reg.promote(v1)
        # Now register and promote v2
        v2 = reg.register(m, "v2", "2023-02-01", "2023-03-01", {}, {}, status="candidate")
        reg.promote(v2)
        assert reg.production_version == v2

        restored = reg.rollback(reason="test rollback")
        assert restored == v1
        assert reg.production_version == v1

    def test_rollback_increments_counter(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v1 = reg.register(m, "v1", "2023-01-01", "2023-02-01", {}, {}, status="candidate")
        reg.promote(v1)
        v2 = reg.register(m, "v2", "2023-02-01", "2023-03-01", {}, {}, status="candidate")
        reg.promote(v2)
        reg.rollback(reason="degraded")
        assert reg.rollback_count == 1

    def test_rollback_marks_degraded_model(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v1 = reg.register(m, "v1", "2023-01-01", "2023-02-01", {}, {}, status="candidate")
        reg.promote(v1)
        v2 = reg.register(m, "v2", "2023-02-01", "2023-03-01", {}, {}, status="candidate")
        reg.promote(v2)
        reg.rollback(reason="post-deployment recall drop")
        assert reg.get_record(v2).status == "rolled_back"

    def test_rollback_without_previous_raises(self, tmp_path):
        reg = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        reg.register(m, "only", "2023-01-01", "2023-02-01", {}, {}, status="production")
        with pytest.raises(RuntimeError):
            reg.rollback(reason="no previous model")


# ============================================================
# Persistence Tests
# ============================================================

class TestModelRegistryPersistence:
    def test_index_reloads(self, tmp_path):
        reg1 = ModelRegistry(models_dir=str(tmp_path))
        m = _make_model()
        v = reg1.register(m, "prod", "2023-01-01", "2023-02-01", {}, {}, status="production")

        # New registry instance reloads from disk
        reg2 = ModelRegistry(models_dir=str(tmp_path))
        assert reg2.production_version == v
        assert len(reg2.get_all_records()) == 1

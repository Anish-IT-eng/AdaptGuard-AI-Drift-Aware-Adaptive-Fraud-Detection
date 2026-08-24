"""
AdaptGuard AI — Baseline Models
Logistic Regression, Random Forest, XGBoost + Periodic Retraining wrapper.
All models share the same sklearn-compatible interface.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Any
from datetime import timedelta

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.logger import get_logger
from src.utils.config import load_config

log = get_logger("models.baseline")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_logistic_regression(cfg: dict) -> LogisticRegression:
    p = cfg["models"]["logistic_regression"]
    return LogisticRegression(
        C=p["C"],
        max_iter=p["max_iter"],
        class_weight=p["class_weight"],
        random_state=p["random_state"],
        solver="lbfgs",
    )


def build_random_forest(cfg: dict) -> RandomForestClassifier:
    p = cfg["models"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        class_weight=p["class_weight"],
        random_state=p["random_state"],
        n_jobs=p["n_jobs"],
    )


def build_xgboost(cfg: dict) -> XGBClassifier:
    p = cfg["models"]["xgboost"]
    return XGBClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        scale_pos_weight=p["scale_pos_weight"],
        random_state=p["random_state"],
        eval_metric=p["eval_metric"],
        early_stopping_rounds=p["early_stopping_rounds"],
        use_label_encoder=False,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Static model wrapper
# ---------------------------------------------------------------------------

class StaticModel:
    """
    Wraps a sklearn/XGBoost model for static (frozen) inference.

    After initial training the model is never updated.
    Represents: "Train once → deploy → no adaptation."
    """

    def __init__(self, model, name: str = "static"):
        self.model   = model
        self.name    = name
        self.version = 1
        self.trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StaticModel":
        log.info(f"[{self.name}] Training on {len(X):,} samples ...")
        self.model.fit(X, y, **kwargs)
        self.trained = True
        log.info(f"[{self.name}] Training complete.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise RuntimeError(f"Model '{self.name}' has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info(f"[{self.name}] Saved to {path}")

    @staticmethod
    def load(path: str) -> "StaticModel":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Periodic Retraining wrapper
# ---------------------------------------------------------------------------

class PeriodicRetrainingModel:
    """
    Retrains the underlying model every N days using the most recent data.

    Represents: "Retrain on a fixed schedule."

    The model keeps a rolling buffer of recent transactions.
    On each retrain trigger, it fits on the last `window_days` of data.
    """

    def __init__(
        self,
        model_factory,
        retrain_interval_days: int = 7,
        window_days: int = 60,
        name: str = "periodic",
    ):
        self.model_factory           = model_factory
        self.retrain_interval_days   = retrain_interval_days
        self.window_days             = window_days
        self.name                    = name
        self.model                   = None
        self.last_retrain_date       = None
        self.version                 = 0
        self.retrain_count           = 0
        self._buffer: list[tuple]    = []          # (datetime, features, label)

    def initial_fit(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> None:
        log.info(f"[{self.name}] Initial training on {len(X):,} samples ...")
        self.model = self.model_factory()
        self.model.fit(X, y)
        self.last_retrain_date = dates.max()
        self.version += 1
        log.info(f"[{self.name}] Initial training done (v{self.version})")

    def observe(
        self,
        X_row: pd.Series,
        y: int,
        current_date: pd.Timestamp,
    ) -> bool:
        """
        Observe a new confirmed-label sample.
        Triggers retraining if interval has passed.

        Returns True if retrain occurred.
        """
        self._buffer.append((current_date, X_row, y))
        # Prune buffer
        cutoff = current_date - timedelta(days=self.window_days)
        self._buffer = [(d, x, l) for (d, x, l) in self._buffer if d >= cutoff]

        if (
            self.last_retrain_date is not None
            and (current_date - self.last_retrain_date).days >= self.retrain_interval_days
            and len(self._buffer) > 50
        ):
            self._retrain(current_date)
            return True
        return False

    def _retrain(self, current_date: pd.Timestamp) -> None:
        dates_buf, X_buf, y_buf = zip(*self._buffer)
        X_df = pd.DataFrame(list(X_buf))
        y_s  = pd.Series(list(y_buf))

        log.info(
            f"[{self.name}] Retraining (v{self.version}→{self.version+1}) "
            f"on {len(X_df):,} samples (interval={self.retrain_interval_days}d)"
        )
        new_model = self.model_factory()
        new_model.fit(X_df, y_s)

        self.model               = new_model
        self.last_retrain_date   = current_date
        self.version            += 1
        self.retrain_count      += 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] Model not initialized.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

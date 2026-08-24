"""
AdaptGuard AI — Online Learning Models
Always-Online learner using SGDClassifier (partial_fit).
Represents: "Update continuously whenever a confirmed label arrives."
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.config import load_config

log = get_logger("models.online")


class AlwaysOnlineModel:
    """
    Online learner that updates on every confirmed label.

    Uses SGDClassifier.partial_fit() to update incrementally.
    Represents the upper-bound adaptation frequency baseline.

    Key difference from AdaptiveML:
    - Updates on EVERY confirmed label (no selectivity)
    - No drift detection
    - No validation gate
    - Can become unstable if labels are noisy
    """

    def __init__(
        self,
        loss: str = "log_loss",
        class_weight: str = "balanced",
        random_state: int = 42,
        name: str = "always_online",
    ):
        self.name = name
        self.model = SGDClassifier(
            loss=loss,
            class_weight=class_weight,
            random_state=random_state,
            learning_rate="optimal",
            max_iter=1,
            warm_start=True,
        )
        self.scaler  = StandardScaler()
        self.version = 1
        self.update_count = 0
        self._fitted = False

    def initial_fit(self, X: pd.DataFrame, y: pd.Series) -> "AlwaysOnlineModel":
        """Initial batch training before streaming begins."""
        log.info(f"[{self.name}] Initial training on {len(X):,} samples ...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, classes=[0, 1])
        self._fitted = True
        log.info(f"[{self.name}] Initial training done.")
        return self

    def observe(self, X_row: pd.DataFrame, y: int) -> None:
        """
        Update the model with a single confirmed-label sample.

        Args:
            X_row: Single-row DataFrame with feature values.
            y:     Confirmed fraud label (0 or 1).
        """
        if not self._fitted:
            raise RuntimeError("Model must be initially fitted before observing.")

        X_scaled = self.scaler.transform(X_row)
        self.model.partial_fit(X_scaled, [y], classes=[0, 1])
        self.update_count += 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def total_updates(self) -> int:
        return self.update_count


class RiverOnlineModel:
    """
    Online model using the River library for streaming ML.
    Wraps River's HoeffdingAdaptiveTreeClassifier.

    River provides native support for concept drift via ADWIN internally.
    Used as an alternative online baseline.

    Note: Requires river>=0.21.0
    """

    def __init__(self, name: str = "river_hat"):
        self.name = name
        self.update_count = 0
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from river import tree
            self._model = tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=200,
                delta=1e-7,
                leaf_prediction="nb",
            )
            log.info(f"[{self.name}] River HoeffdingAdaptiveTreeClassifier initialized.")
        except ImportError:
            log.error("River library not installed. Install with: pip install river==0.21.2")
            self._model = None

    def observe(self, x: dict, y: int) -> None:
        """Update with a single observation (x as dict, y as label)."""
        if self._model is None:
            return
        self._model.learn_one(x, y)
        self.update_count += 1

    def predict_proba_one(self, x: dict) -> float:
        """Predict fraud probability for one transaction (x as dict)."""
        if self._model is None:
            return 0.5
        proba = self._model.predict_proba_one(x)
        return proba.get(1, 0.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([
            self.predict_proba_one(row.to_dict())
            for _, row in X.iterrows()
        ])

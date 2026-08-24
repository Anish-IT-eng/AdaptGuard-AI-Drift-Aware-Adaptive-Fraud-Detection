"""
AdaptGuard AI — Data Drift Monitor (Label-Free Channel)
Detects changes in P(X) using PSI, KS Test, and MMD.

IMPORTANT: This monitor fires WITHOUT waiting for labels.
It can detect feature distribution changes immediately from transaction data.

Architecture role: First drift signal — earliest warning of environmental change.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Optional
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.utils.config import load_config

log = get_logger("drift.data_monitor")


@dataclass
class DataDriftResult:
    """Result from one data drift monitoring cycle."""
    timestamp:       pd.Timestamp
    psi_scores:      dict[str, float] = field(default_factory=dict)
    ks_scores:       dict[str, float] = field(default_factory=dict)
    mmd_score:       Optional[float]  = None
    max_psi:         float = 0.0
    n_features_alert: int  = 0
    drift_detected:  bool  = False
    summary:         str   = ""


class PSIMonitor:
    """
    Population Stability Index monitor for feature-level distribution shift.

    PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

    NOTE: Thresholds 0.10 (warn) and 0.20 (alert) are starting points only.
    These MUST be calibrated experimentally for this specific dataset.
    Generic financial-score PSI thresholds cannot be assumed universal.
    """

    def __init__(
        self,
        warn_threshold:  float = 0.10,
        alert_threshold: float = 0.20,
        n_bins:          int   = 10,
        min_bucket_size: int   = 5,
    ):
        self.warn_threshold  = warn_threshold
        self.alert_threshold = alert_threshold
        self.n_bins          = n_bins
        self.min_bucket_size = min_bucket_size
        self._reference: Optional[dict[str, np.ndarray]] = None

    def set_reference(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """
        Compute reference distribution from the initial training window.
        Must be called before compute_psi().
        """
        self._reference = {}
        for col in feature_cols:
            if col in df.columns:
                values = df[col].dropna().values
                if len(values) == 0:
                    continue
                bins = np.nanpercentile(values, np.linspace(0, 100, self.n_bins + 1))
                bins = np.unique(bins)
                hist, _ = np.histogram(values, bins=bins)
                hist = np.where(hist < self.min_bucket_size, self.min_bucket_size, hist)
                self._reference[col] = (bins, hist / hist.sum())

        log.info(f"PSI reference set for {len(self._reference)} features.")

    def compute_psi(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Compute PSI for all reference features on current window.

        Returns dict mapping feature_name → PSI score.
        """
        if self._reference is None:
            raise RuntimeError("Reference distribution not set. Call set_reference() first.")

        psi_scores = {}
        for col, (bins, ref_pct) in self._reference.items():
            if col not in df.columns:
                continue
            values = df[col].dropna().values
            if len(values) == 0:
                psi_scores[col] = 0.0
                continue

            actual_hist, _ = np.histogram(values, bins=bins)
            actual_hist = np.where(actual_hist == 0, 1e-6, actual_hist)
            actual_pct  = actual_hist / actual_hist.sum()

            ref_safe = np.where(ref_pct == 0, 1e-6, ref_pct)
            psi = np.sum((actual_pct - ref_safe) * np.log(actual_pct / ref_safe + 1e-10))
            psi_scores[col] = float(psi)

        return psi_scores

    def is_alert(self, psi: float) -> bool:
        return psi >= self.alert_threshold

    def is_warn(self, psi: float) -> bool:
        return psi >= self.warn_threshold


class KSMonitor:
    """
    Kolmogorov-Smirnov test monitor for continuous feature distribution shifts.
    Label-free — compares feature distributions only.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha     = alpha
        self._reference_data: Optional[dict[str, np.ndarray]] = None

    def set_reference(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        self._reference_data = {
            col: df[col].dropna().values
            for col in feature_cols if col in df.columns
        }
        log.info(f"KS reference set for {len(self._reference_data)} features.")

    def compute_ks(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Returns dict mapping feature_name → KS p-value.
        Low p-value (<alpha) indicates significant distributional shift.
        """
        if self._reference_data is None:
            raise RuntimeError("Reference not set.")

        ks_scores = {}
        for col, ref_vals in self._reference_data.items():
            if col not in df.columns or len(ref_vals) == 0:
                continue
            current_vals = df[col].dropna().values
            if len(current_vals) == 0:
                continue
            stat, p_value = stats.ks_2samp(ref_vals, current_vals)
            ks_scores[col] = float(p_value)

        return ks_scores

    def n_drifted(self, ks_scores: dict[str, float]) -> int:
        """Count features with significant KS drift (p < alpha)."""
        return sum(1 for p in ks_scores.values() if p < self.alpha)


class MMDMonitor:
    """
    Maximum Mean Discrepancy monitor using an RBF kernel.
    Multivariate — detects silent drift across feature space jointly.

    Used as an advanced experiment (Phase 12). More computationally intensive.
    """

    def __init__(self, gamma: float = 1.0, n_permutations: int = 100, alpha: float = 0.05):
        self.gamma          = gamma
        self.n_permutations = n_permutations
        self.alpha          = alpha
        self._reference_data: Optional[np.ndarray] = None

    def set_reference(self, X: np.ndarray) -> None:
        self._reference_data = X.copy()
        log.info(f"MMD reference set: shape={X.shape}")

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> float:
        dists  = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-self.gamma * dists).mean()

    def compute_mmd(self, X_current: np.ndarray, max_samples: int = 500) -> float:
        """
        Compute unbiased MMD² estimate between reference and current.
        Subsamples if needed for performance.
        """
        if self._reference_data is None:
            raise RuntimeError("Reference not set.")

        X_ref = self._reference_data
        if len(X_ref) > max_samples:
            idx   = np.random.choice(len(X_ref), max_samples, replace=False)
            X_ref = X_ref[idx]
        if len(X_current) > max_samples:
            idx       = np.random.choice(len(X_current), max_samples, replace=False)
            X_current = X_current[idx]

        mmd2 = (
            self._rbf_kernel(X_ref, X_ref)
            + self._rbf_kernel(X_current, X_current)
            - 2 * self._rbf_kernel(X_ref, X_current)
        )
        return float(max(mmd2, 0.0))


# ---------------------------------------------------------------------------
# Unified Data Drift Monitor
# ---------------------------------------------------------------------------

class DataDriftMonitor:
    """
    Orchestrates PSI, KS, and MMD monitors.
    Called on each streaming batch WITHOUT requiring labels.
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or load_config()
        dc  = cfg["drift"]

        self.psi = PSIMonitor(
            warn_threshold  = dc["psi"]["warn_threshold"],
            alert_threshold = dc["psi"]["alert_threshold"],
            n_bins          = dc["psi"]["bins"],
            min_bucket_size = dc["psi"]["min_bucket_size"],
        )
        self.ks  = KSMonitor(alpha=0.05)
        self.mmd = MMDMonitor(gamma=1.0, n_permutations=dc["mmd"]["n_permutations"])

        self.feature_cols: list[str] = []
        self.history: list[DataDriftResult] = []

    def initialize(self, reference_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Set reference distributions from initial training window."""
        self.feature_cols = feature_cols
        self.psi.set_reference(reference_df, feature_cols)
        self.ks.set_reference(reference_df, feature_cols)

        ref_arr = reference_df[feature_cols].fillna(0).values
        self.mmd.set_reference(ref_arr)
        log.info("DataDriftMonitor initialized with reference distributions.")

    def monitor(
        self,
        current_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
        run_mmd: bool = False,
    ) -> DataDriftResult:
        """
        Run one monitoring cycle on the current data window.

        Args:
            current_df: Current transaction batch (no labels required).
            timestamp:  Current logical time.
            run_mmd:    Whether to run the more expensive MMD test.

        Returns:
            DataDriftResult with all scores.
        """
        ts = timestamp or pd.Timestamp.now()

        psi_scores = self.psi.compute_psi(current_df)
        ks_scores  = self.ks.compute_ks(current_df)

        mmd_score = None
        if run_mmd and len(current_df) > 10:
            X_curr    = current_df[self.feature_cols].fillna(0).values
            mmd_score = self.mmd.compute_mmd(X_curr)

        max_psi          = max(psi_scores.values(), default=0.0)
        n_features_alert = sum(1 for p in psi_scores.values() if self.psi.is_alert(p))
        n_ks_drifted     = self.ks.n_drifted(ks_scores)

        drift_detected = (
            n_features_alert > 0
            or n_ks_drifted > int(0.2 * len(ks_scores))
            or (mmd_score is not None and mmd_score > 0.05)
        )

        summary = (
            f"PSI max={max_psi:.4f} ({n_features_alert} features alert) | "
            f"KS drifted={n_ks_drifted}/{len(ks_scores)} | "
            f"MMD={mmd_score:.4f if mmd_score else 'N/A'}"
        )

        result = DataDriftResult(
            timestamp        = ts,
            psi_scores       = psi_scores,
            ks_scores        = ks_scores,
            mmd_score        = mmd_score,
            max_psi          = max_psi,
            n_features_alert = n_features_alert,
            drift_detected   = drift_detected,
            summary          = summary,
        )

        if drift_detected:
            log.warning(f"[DataDriftMonitor] DRIFT DETECTED: {summary}")
        else:
            log.debug(f"[DataDriftMonitor] Stable: {summary}")

        self.history.append(result)
        return result

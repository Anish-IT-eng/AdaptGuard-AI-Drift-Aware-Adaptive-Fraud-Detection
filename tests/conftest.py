"""
AdaptGuard AI — pytest shared fixtures.
Provides lightweight synthetic data for all unit tests.
No real simulation is run — fixtures are deterministic and fast.
"""

import pytest
import numpy as np
import pandas as pd


# ============================================================
# Tiny synthetic transaction fixture (no simulator needed)
# ============================================================

@pytest.fixture
def small_tx_df():
    """200-row synthetic transaction DataFrame."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "TRANSACTION_ID":    range(n),
        "TX_DATETIME":       dates,
        "CUSTOMER_ID":       np.random.randint(0, 50, n),
        "TERMINAL_ID":       np.random.randint(0, 100, n),
        "TX_AMOUNT":         np.random.uniform(5, 200, n),
        "TX_FRAUD":          (np.random.rand(n) < 0.008).astype(int),
    })
    return df


@pytest.fixture
def feature_df(small_tx_df):
    """Small DataFrame with basic engineered features."""
    df = small_tx_df.copy()
    df["TX_HOUR"]        = df["TX_DATETIME"].dt.hour
    df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek
    df["TX_IS_WEEKEND"]  = (df["TX_DAY_OF_WEEK"] >= 5).astype(int)
    df["TX_AMOUNT_LOG"]  = np.log1p(df["TX_AMOUNT"])
    return df


@pytest.fixture
def feature_cols():
    return ["TX_AMOUNT", "TX_HOUR", "TX_DAY_OF_WEEK", "TX_IS_WEEKEND", "TX_AMOUNT_LOG"]


@pytest.fixture
def binary_labels():
    """y_true / y_pred / y_proba arrays for metric tests."""
    np.random.seed(0)
    n = 500
    y_true  = (np.random.rand(n) < 0.05).astype(int)
    y_proba = np.clip(y_true * 0.7 + np.random.rand(n) * 0.3, 0, 1)
    y_pred  = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba

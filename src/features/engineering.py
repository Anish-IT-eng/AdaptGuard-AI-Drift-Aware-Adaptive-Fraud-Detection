"""
AdaptGuard AI — Leak-Free Feature Engineering
All behavioral features use ONLY historical data available before transaction time t.

Critical rule: feature_information_time < transaction_time
No global means or aggregates over the full dataset.
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger("features.engineering")


# ---------------------------------------------------------------------------
# Time Features (no leakage — derived from TX_DATETIME only)
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and time-based features from the transaction timestamp."""
    df = df.copy()
    dt = df["TX_DATETIME"]

    df["TX_HOUR"]        = dt.dt.hour
    df["TX_DAY"]         = dt.dt.day
    df["TX_DAY_OF_WEEK"] = dt.dt.dayofweek          # 0=Monday, 6=Sunday
    df["TX_IS_WEEKEND"]  = (dt.dt.dayofweek >= 5).astype(int)
    df["TX_MONTH"]       = dt.dt.month
    df["TX_DAY_OF_YEAR"] = dt.dt.dayofyear

    # Time since previous transaction (seconds) — per customer, shift(1) prevents leakage
    df = df.sort_values("TX_DATETIME")
    df["TX_PREV_DATETIME"] = df.groupby("CUSTOMER_ID")["TX_DATETIME"].shift(1)
    df["TX_TIME_SINCE_PREV"] = (
        df["TX_DATETIME"] - df["TX_PREV_DATETIME"]
    ).dt.total_seconds().fillna(-1)

    df.drop(columns=["TX_PREV_DATETIME"], inplace=True)
    log.debug("Time features added.")
    return df


# ---------------------------------------------------------------------------
# Customer Behavioral Features (rolling — strictly historical)
# ---------------------------------------------------------------------------

def add_customer_features(
    df: pd.DataFrame,
    windows_days: list[int] = [1, 7, 30],
) -> pd.DataFrame:
    """
    Add customer-level rolling behavioral features.

    All aggregates are computed over the historical window ending
    BEFORE the current transaction (shift applied to prevent leakage).

    Features per window W:
    - customer_nb_tx_{W}day:   transaction count in last W days
    - customer_avg_amount_{W}day: average amount in last W days
    - customer_std_amount_{W}day: std amount in last W days
    - customer_sum_amount_{W}day: total amount in last W days
    """
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)

    for w in windows_days:
        window_str = f"{w}day"
        nb_col  = f"CUSTOMER_NB_TX_{window_str}"
        avg_col = f"CUSTOMER_AVG_AMOUNT_{window_str}"
        std_col = f"CUSTOMER_STD_AMOUNT_{window_str}"
        sum_col = f"CUSTOMER_SUM_AMOUNT_{window_str}"

        df[nb_col]  = 0.0
        df[avg_col] = 0.0
        df[std_col] = 0.0
        df[sum_col] = 0.0

        for cust_id, group in df.groupby("CUSTOMER_ID"):
            group = group.sort_values("TX_DATETIME")
            for i, (idx, row) in enumerate(group.iterrows()):
                cutoff = row["TX_DATETIME"] - pd.Timedelta(days=w)
                # Historical transactions only — strictly before current
                hist = group.iloc[:i]
                hist = hist[hist["TX_DATETIME"] >= cutoff]

                df.at[idx, nb_col]  = len(hist)
                df.at[idx, avg_col] = hist["TX_AMOUNT"].mean() if len(hist) > 0 else 0.0
                df.at[idx, std_col] = hist["TX_AMOUNT"].std()  if len(hist) > 1 else 0.0
                df.at[idx, sum_col] = hist["TX_AMOUNT"].sum()  if len(hist) > 0 else 0.0

        log.debug(f"Customer features added for window={w}d")

    # Amount deviation from historical mean (7d)
    if "CUSTOMER_AVG_AMOUNT_7day" in df.columns:
        df["CUSTOMER_AMOUNT_DEVIATION"] = (
            df["TX_AMOUNT"] - df["CUSTOMER_AVG_AMOUNT_7day"]
        ).fillna(0.0)

    return df


def add_customer_features_fast(
    df: pd.DataFrame,
    windows_days: list[int] = [1, 7, 30],
) -> pd.DataFrame:
    """
    Faster vectorized customer feature computation using expanding window.
    Uses shift(1) to ensure no current-transaction leakage.

    Note: This is an approximation for efficiency on large datasets.
    The exact rolling window by time-delta can be used for smaller sets.
    """
    df = df.sort_values(["CUSTOMER_ID", "TX_DATETIME"]).copy()

    for w in windows_days:
        window_str = f"{w}day"
        win_size   = int(w * 10)  # approximate: assume ~10 tx/day average

        grouped = df.groupby("CUSTOMER_ID")["TX_AMOUNT"]

        df[f"CUSTOMER_NB_TX_{window_str}"] = (
            grouped.transform(lambda x: x.shift(1).rolling(window=win_size, min_periods=1).count())
        ).fillna(0)

        df[f"CUSTOMER_AVG_AMOUNT_{window_str}"] = (
            grouped.transform(lambda x: x.shift(1).rolling(window=win_size, min_periods=1).mean())
        ).fillna(0)

        df[f"CUSTOMER_STD_AMOUNT_{window_str}"] = (
            grouped.transform(lambda x: x.shift(1).rolling(window=win_size, min_periods=1).std())
        ).fillna(0)

        df[f"CUSTOMER_SUM_AMOUNT_{window_str}"] = (
            grouped.transform(lambda x: x.shift(1).rolling(window=win_size, min_periods=1).sum())
        ).fillna(0)

    df["CUSTOMER_AMOUNT_DEVIATION"] = (
        df["TX_AMOUNT"] - df.get("CUSTOMER_AVG_AMOUNT_7day", 0)
    ).fillna(0)

    log.debug("Customer features (fast) added.")
    return df


# ---------------------------------------------------------------------------
# Terminal Behavioral Features (rolling — strictly historical)
# ---------------------------------------------------------------------------

def add_terminal_features(
    df: pd.DataFrame,
    windows_days: list[int] = [1, 7, 30],
) -> pd.DataFrame:
    """
    Add terminal-level rolling behavioral features.

    For terminal fraud rate: uses only fraud labels that were available
    before the current transaction (shift(1) applied).
    """
    df = df.sort_values(["TERMINAL_ID", "TX_DATETIME"]).copy()

    for w in windows_days:
        window_str = f"{w}day"
        win_size   = int(w * 50)  # approximate

        term_amount = df.groupby("TERMINAL_ID")["TX_AMOUNT"]
        term_fraud  = df.groupby("TERMINAL_ID")["TX_FRAUD"]

        df[f"TERMINAL_NB_TX_{window_str}"] = (
            term_amount.transform(lambda x: x.shift(1).rolling(win_size, min_periods=1).count())
        ).fillna(0)

        df[f"TERMINAL_AVG_AMOUNT_{window_str}"] = (
            term_amount.transform(lambda x: x.shift(1).rolling(win_size, min_periods=1).mean())
        ).fillna(0)

        # Historical fraud rate — uses only previously confirmed labels
        df[f"TERMINAL_FRAUD_RATE_{window_str}"] = (
            term_fraud.transform(lambda x: x.shift(1).rolling(win_size, min_periods=1).mean())
        ).fillna(0)

    df["TERMINAL_AMOUNT_DEVIATION"] = (
        df["TX_AMOUNT"] - df.get("TERMINAL_AVG_AMOUNT_7day", 0)
    ).fillna(0)

    log.debug("Terminal features added.")
    return df


# ---------------------------------------------------------------------------
# Velocity Features (short time-window counts)
# ---------------------------------------------------------------------------

def add_velocity_features(
    df: pd.DataFrame,
    windows_minutes: list[int] = [5, 10, 60, 1440],
) -> pd.DataFrame:
    """
    Add transaction velocity features per customer.

    For each window W minutes:
    - VELOCITY_NB_TX_{W}min:     # of transactions in last W min (excluding current)
    - VELOCITY_SUM_AMOUNT_{W}min: total amount in last W min

    These use strictly historical information (before current TX_DATETIME).
    """
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)

    for w in windows_minutes:
        window_label = f"{w}min" if w < 1440 else "24h"
        nb_col  = f"VELOCITY_NB_TX_{window_label}"
        sum_col = f"VELOCITY_SUM_AMOUNT_{window_label}"

        df[nb_col]  = 0
        df[sum_col] = 0.0

    tx_by_customer = {}
    for idx, row in df.iterrows():
        cid = row["CUSTOMER_ID"]
        t   = row["TX_DATETIME"]

        if cid not in tx_by_customer:
            tx_by_customer[cid] = []

        for w in windows_minutes:
            window_label = f"{w}min" if w < 1440 else "24h"
            nb_col  = f"VELOCITY_NB_TX_{window_label}"
            sum_col = f"VELOCITY_SUM_AMOUNT_{window_label}"

            cutoff = t - pd.Timedelta(minutes=w)
            history = [
                (ht, ha) for (ht, ha) in tx_by_customer[cid]
                if ht >= cutoff   # strictly historical (before current insert)
            ]
            df.at[idx, nb_col]  = len(history)
            df.at[idx, sum_col] = sum(ha for _, ha in history)

        # Append AFTER processing (no leakage)
        tx_by_customer[cid].append((t, row["TX_AMOUNT"]))

        # Prune buffer to last 24h to save memory
        cutoff_24h = t - pd.Timedelta(minutes=1440)
        tx_by_customer[cid] = [
            (ht, ha) for (ht, ha) in tx_by_customer[cid] if ht >= cutoff_24h
        ]

    log.debug("Velocity features added.")
    return df


# ---------------------------------------------------------------------------
# Master Feature Builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    use_fast_customer_features: bool = True,
) -> pd.DataFrame:
    """
    Build all features on a validated, chronologically sorted DataFrame.

    Args:
        df: Validated transaction DataFrame.
        use_fast_customer_features: Use vectorized (fast) customer features.

    Returns:
        DataFrame with all engineered features.
    """
    log.info(f"Building features for {len(df):,} transactions ...")

    df = add_time_features(df)

    if use_fast_customer_features:
        df = add_customer_features_fast(df, windows_days=[1, 7, 30])
    else:
        df = add_customer_features(df, windows_days=[1, 7, 30])

    df = add_terminal_features(df, windows_days=[1, 7, 30])
    df = add_velocity_features(df, windows_minutes=[5, 10, 60, 1440])

    log.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of engineered feature column names (excludes IDs and labels)."""
    exclude = {
        "TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID",
        "TX_FRAUD", "TX_FRAUD_SCENARIO", "TX_DRIFT_INJECTED",
        "x_customer_id", "y_customer_id", "x_terminal_id", "y_terminal_id",
        "mean_amount", "std_amount", "mean_nb_tx_per_day", "available_terminals",
    }
    return [c for c in df.columns if c not in exclude]

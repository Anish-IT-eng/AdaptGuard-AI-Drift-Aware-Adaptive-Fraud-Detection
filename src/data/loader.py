"""
AdaptGuard AI — Data Loader
Loads raw or processed transaction data with temporal integrity checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.data.validator import DataValidator

log = get_logger("data.loader")


def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw CSV transaction data.

    Args:
        path: Path to CSV file.

    Returns:
        Validated, chronologically sorted DataFrame.
    """
    log.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path, parse_dates=["TX_DATETIME"])
    validator = DataValidator()
    df, report = validator.validate(df)
    log.info(f"Loaded: {len(df):,} rows")
    return df


def load_processed(path: str) -> pd.DataFrame:
    """Load feature-engineered parquet file."""
    log.info(f"Loading processed data from: {path}")
    df = pd.read_parquet(path)
    # Ensure chronological order
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)
    log.info(f"Loaded: {len(df):,} rows | Columns: {len(df.columns)}")
    return df


def chronological_split(
    df: pd.DataFrame,
    train_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and evaluation portions by time.
    NO shuffling — strictly chronological.

    Args:
        df:         Full transaction DataFrame.
        train_days: Number of days for initial training.

    Returns:
        (train_df, eval_df)
    """
    from datetime import timedelta
    start = df["TX_DATETIME"].min()
    split_date = start + timedelta(days=train_days)

    train_df = df[df["TX_DATETIME"] < split_date].copy()
    eval_df  = df[df["TX_DATETIME"] >= split_date].copy()

    log.info(
        f"Chronological split at {split_date}: "
        f"train={len(train_df):,} | eval={len(eval_df):,}"
    )
    return train_df, eval_df


def get_window(
    df: pd.DataFrame,
    end_date: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    """
    Extract a time window ending at end_date.
    Used for candidate training and validation windows.
    """
    start = end_date - pd.Timedelta(days=window_days)
    return df[(df["TX_DATETIME"] >= start) & (df["TX_DATETIME"] < end_date)].copy()

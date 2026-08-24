"""
AdaptGuard AI — Data Validator
Checks temporal integrity and data quality without leaking future information.

Key rules from the spec:
- Dataset must remain chronological
- No global shuffling
- Timestamp gaps are allowed (natural in real streams)
- Duplicate TX IDs: detect, investigate, remove only confirmed duplicates
- Amounts must be positive
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from src.utils.logger import get_logger

log = get_logger("data.validator")


class DataValidator:
    """
    Validates the raw transaction dataset.

    Checks performed:
    1. Required columns present
    2. Missing values
    3. Amount validity (positive)
    4. Timestamp format and chronological ordering
    5. Duplicate transaction detection (not auto-deletion)
    6. Label integrity
    """

    REQUIRED_COLUMNS = [
        "TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID",
        "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD",
    ]

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Run all validation checks.

        Args:
            df: Raw transaction DataFrame.

        Returns:
            (cleaned_df, report) where report contains all findings.
        """
        report = {}
        df = df.copy()

        log.info(f"Starting validation: {len(df):,} rows")

        df, report = self._check_columns(df, report)
        df, report = self._check_missing(df, report)
        df, report = self._check_amounts(df, report)
        df, report = self._check_timestamps(df, report)
        df, report = self._check_duplicates(df, report)
        df, report = self._check_labels(df, report)
        df, report = self._sort_chronological(df, report)

        log.info(
            f"Validation complete. "
            f"Rows remaining: {len(df):,} | "
            f"Issues found: {report.get('total_issues', 0)}"
        )
        return df, report

    # ------------------------------------------------------------------
    def _check_columns(self, df, report):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        report["columns_ok"] = True
        log.debug("Column check passed.")
        return df, report

    def _check_missing(self, df, report):
        mv = df[self.REQUIRED_COLUMNS].isnull().sum()
        mv_dict = mv[mv > 0].to_dict()
        report["missing_values"] = mv_dict

        if mv_dict:
            log.warning(f"Missing values found: {mv_dict}")
            # Drop rows with missing values in critical columns
            critical = ["TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD"]
            before = len(df)
            df = df.dropna(subset=critical)
            report["rows_dropped_missing"] = before - len(df)
            log.info(f"  Dropped {before - len(df):,} rows with critical missing values")
        else:
            report["rows_dropped_missing"] = 0
            log.debug("No missing values in critical columns.")

        return df, report

    def _check_amounts(self, df, report):
        negative_mask = df["TX_AMOUNT"] <= 0
        n_negative = negative_mask.sum()
        report["negative_amounts"] = int(n_negative)

        if n_negative > 0:
            log.warning(f"  {n_negative:,} transactions with non-positive amounts — removing")
            df = df[~negative_mask]
        else:
            log.debug("Amount check passed — all positive.")

        # Flag extreme outliers (> 99.9th percentile) for investigation
        upper = df["TX_AMOUNT"].quantile(0.999)
        n_extreme = int((df["TX_AMOUNT"] > upper).sum())
        report["extreme_amount_outliers"] = n_extreme
        if n_extreme > 0:
            log.info(f"  {n_extreme:,} transactions above 99.9th percentile (${upper:.2f}) — flagged for review")

        return df, report

    def _check_timestamps(self, df, report):
        # Parse timestamps if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df["TX_DATETIME"]):
            try:
                df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
            except Exception as e:
                raise ValueError(f"Cannot parse TX_DATETIME: {e}")

        # Check for NaT after parsing
        n_nat = df["TX_DATETIME"].isna().sum()
        report["invalid_timestamps"] = int(n_nat)
        if n_nat > 0:
            log.warning(f"  {n_nat:,} unparseable timestamps — removing")
            df = df.dropna(subset=["TX_DATETIME"])

        # Verify chronological ordering (natural gaps are OK — do NOT require gap-free)
        is_sorted = df["TX_DATETIME"].is_monotonic_increasing
        report["timestamps_sorted"] = bool(is_sorted)
        if not is_sorted:
            log.info("  Timestamps not sorted — will sort in next step")

        report["date_range"] = {
            "min": str(df["TX_DATETIME"].min()),
            "max": str(df["TX_DATETIME"].max()),
            "days": (df["TX_DATETIME"].max() - df["TX_DATETIME"].min()).days,
        }
        log.info(
            f"  Date range: {report['date_range']['min']} → "
            f"{report['date_range']['max']} "
            f"({report['date_range']['days']} days)"
        )
        return df, report

    def _check_duplicates(self, df, report):
        """
        Detect duplicate TRANSACTION_IDs.
        Per spec: detect → report → remove only CONFIRMED duplicates.
        A confirmed duplicate is a row with identical TX_DATETIME, CUSTOMER_ID,
        TERMINAL_ID, and TX_AMOUNT to another row.
        """
        dup_id_mask = df["TRANSACTION_ID"].duplicated(keep=False)
        n_dup_ids   = dup_id_mask.sum()
        report["duplicate_transaction_ids"] = int(n_dup_ids)

        if n_dup_ids > 0:
            log.warning(f"  {n_dup_ids:,} rows share a TRANSACTION_ID — investigating ...")

            # Check if the entire row is a duplicate (confirmed duplicate)
            key_cols = ["TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]
            confirmed_dup_mask = df.duplicated(subset=key_cols, keep="first")
            n_confirmed = confirmed_dup_mask.sum()
            report["confirmed_duplicates_removed"] = int(n_confirmed)

            if n_confirmed > 0:
                log.info(f"  Removing {n_confirmed:,} confirmed full-row duplicates")
                df = df[~confirmed_dup_mask]
            else:
                log.info("  Duplicate IDs found but rows differ — keeping all, manual review needed")
        else:
            report["confirmed_duplicates_removed"] = 0
            log.debug("No duplicate TRANSACTION_IDs.")

        return df, report

    def _check_labels(self, df, report):
        label_counts = df["TX_FRAUD"].value_counts().to_dict()
        report["label_distribution"] = {int(k): int(v) for k, v in label_counts.items()}

        fraud_rate = df["TX_FRAUD"].mean()
        report["fraud_rate"] = float(fraud_rate)
        log.info(
            f"  Labels: {label_counts} | "
            f"Fraud rate: {fraud_rate*100:.3f}%"
        )

        invalid_labels = ~df["TX_FRAUD"].isin([0, 1])
        n_invalid = invalid_labels.sum()
        report["invalid_labels"] = int(n_invalid)
        if n_invalid > 0:
            log.warning(f"  {n_invalid:,} rows with invalid TX_FRAUD values — removing")
            df = df[~invalid_labels]

        return df, report

    def _sort_chronological(self, df, report):
        """Sort by TX_DATETIME — must remain chronological throughout pipeline."""
        df = df.sort_values("TX_DATETIME").reset_index(drop=True)
        report["final_row_count"] = len(df)
        report["total_issues"] = sum([
            report.get("rows_dropped_missing", 0),
            report.get("negative_amounts", 0),
            report.get("invalid_timestamps", 0),
            report.get("confirmed_duplicates_removed", 0),
            report.get("invalid_labels", 0),
        ])
        log.info(f"  Sorted chronologically. Final row count: {len(df):,}")
        return df, report

    def print_report(self, report: dict) -> None:
        """Pretty print the validation report."""
        print("\n" + "=" * 60)
        print("  DATA VALIDATION REPORT")
        print("=" * 60)
        for k, v in report.items():
            print(f"  {k:<40} {v}")
        print("=" * 60 + "\n")

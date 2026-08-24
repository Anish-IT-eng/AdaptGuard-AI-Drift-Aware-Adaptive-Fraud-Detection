"""
Unit tests — src/data/simulator.py
Validates simulator output schema and basic statistical properties.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.simulator import (
    generate_customer_profiles,
    generate_terminal_profiles,
    generate_dataset,
)


class TestCustomerProfiles:
    def test_shape(self):
        df = generate_customer_profiles(n_customers=100, random_state=42)
        assert len(df) == 100

    def test_columns(self):
        df = generate_customer_profiles(n_customers=10)
        expected = {"CUSTOMER_ID", "x_customer_id", "y_customer_id",
                    "mean_amount", "std_amount", "mean_nb_tx_per_day"}
        assert expected.issubset(set(df.columns))

    def test_no_nulls(self):
        df = generate_customer_profiles(n_customers=50)
        assert df.isnull().sum().sum() == 0

    def test_amount_positive(self):
        df = generate_customer_profiles(n_customers=50)
        assert (df["mean_amount"] > 0).all()
        assert (df["std_amount"] > 0).all()

    def test_reproducible(self):
        df1 = generate_customer_profiles(n_customers=20, random_state=7)
        df2 = generate_customer_profiles(n_customers=20, random_state=7)
        pd.testing.assert_frame_equal(df1, df2)


class TestTerminalProfiles:
    def test_shape(self):
        df = generate_terminal_profiles(n_terminals=50, random_state=42)
        assert len(df) == 50

    def test_columns(self):
        df = generate_terminal_profiles(n_terminals=10)
        assert "TERMINAL_ID" in df.columns

    def test_coordinates_in_range(self):
        df = generate_terminal_profiles(n_terminals=100)
        assert (df["x_terminal_id"] >= 0).all()
        assert (df["x_terminal_id"] <= 100).all()


class TestGenerateDataset:
    """
    Smoke test for generate_dataset.
    Uses a tiny configuration to avoid long runtimes in CI.
    """

    def test_output_schema(self, tmp_path):
        """Dataset must have required columns and correct dtypes."""
        out_path = str(tmp_path / "transactions.csv")
        df = generate_dataset(
            n_customers  = 20,
            n_terminals  = 40,
            nb_days      = 5,
            start_date   = "2023-01-01",
            random_state = 42,
            output_path  = out_path,
        )

        required_cols = {
            "TRANSACTION_ID", "TX_DATETIME",
            "CUSTOMER_ID", "TERMINAL_ID",
            "TX_AMOUNT", "TX_FRAUD",
        }
        assert required_cols.issubset(set(df.columns)), (
            f"Missing columns: {required_cols - set(df.columns)}"
        )

    def test_fraud_rate_reasonable(self, tmp_path):
        """Fraud rate should be low (~0.5%–5%) for realism."""
        out_path = str(tmp_path / "transactions.csv")
        df = generate_dataset(
            n_customers=30, n_terminals=60, nb_days=10,
            start_date="2023-01-01", random_state=42,
            output_path=out_path,
        )
        fraud_rate = df["TX_FRAUD"].mean()
        assert 0.0 < fraud_rate < 0.10, (
            f"Fraud rate {fraud_rate:.4f} is outside expected range [0, 0.10]"
        )

    def test_chronological_order(self, tmp_path):
        """Transactions must be in ascending datetime order."""
        out_path = str(tmp_path / "transactions.csv")
        df = generate_dataset(
            n_customers=20, n_terminals=40, nb_days=5,
            start_date="2023-01-01", random_state=42,
            output_path=out_path,
        )
        dts = df["TX_DATETIME"].values
        assert (dts[1:] >= dts[:-1]).all(), "Dataset is not chronologically sorted"

    def test_no_negative_amounts(self, tmp_path):
        out_path = str(tmp_path / "transactions.csv")
        df = generate_dataset(
            n_customers=20, n_terminals=40, nb_days=5,
            start_date="2023-01-01", random_state=42,
            output_path=out_path,
        )
        assert (df["TX_AMOUNT"] >= 0).all()

    def test_output_file_created(self, tmp_path):
        out_path = str(tmp_path / "transactions.csv")
        generate_dataset(
            n_customers=20, n_terminals=40, nb_days=5,
            start_date="2023-01-01", random_state=42,
            output_path=out_path,
        )
        assert Path(out_path).exists()

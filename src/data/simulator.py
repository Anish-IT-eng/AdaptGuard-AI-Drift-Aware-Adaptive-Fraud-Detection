"""
AdaptGuard AI — ULB/Worldline Simulator Interface
Wraps the Fraud Detection Handbook simulator to generate synthetic
time-dependent transaction streams with configurable fraud scenarios.

Reference:
  https://fraud-detection-handbook.github.io/fraud-detection-handbook/
  Chapter_3_GettingStarted/SimulatedDataset.html
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import load_config

log = get_logger("data.simulator")


# ---------------------------------------------------------------------------
# Core transaction generator (based on FDH Handbook methodology)
# ---------------------------------------------------------------------------

def generate_customer_profiles(
    n_customers: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic customer profiles.

    Each customer has:
    - mean_amount: their typical transaction amount
    - std_amount:  variability in amounts
    - mean_nb_tx_per_day: average daily transaction frequency
    """
    rng = np.random.RandomState(random_state)

    customer_id_properties = []
    for customer_id in range(n_customers):
        x_customer = rng.uniform(0, 100)
        y_customer = rng.uniform(0, 100)

        mean_amount = rng.uniform(5, 100)
        std_amount  = mean_amount / 2

        mean_nb_tx_per_day = rng.uniform(0, 4)

        customer_id_properties.append([
            customer_id,
            x_customer, y_customer,
            mean_amount, std_amount,
            mean_nb_tx_per_day,
        ])

    return pd.DataFrame(
        customer_id_properties,
        columns=[
            "CUSTOMER_ID",
            "x_customer_id", "y_customer_id",
            "mean_amount", "std_amount",
            "mean_nb_tx_per_day",
        ],
    )


def generate_terminal_profiles(
    n_terminals: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic terminal (merchant) profiles."""
    rng = np.random.RandomState(random_state + 1)

    terminal_id_properties = []
    for terminal_id in range(n_terminals):
        x_terminal = rng.uniform(0, 100)
        y_terminal = rng.uniform(0, 100)
        terminal_id_properties.append([terminal_id, x_terminal, y_terminal])

    return pd.DataFrame(
        terminal_id_properties,
        columns=["TERMINAL_ID", "x_terminal_id", "y_terminal_id"],
    )


def associate_customers_to_terminals(
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    radius: float = 5.0,
) -> pd.DataFrame:
    """
    Each customer is associated with terminals within a geographic radius.
    Returns customer_profiles with an added 'available_terminals' column.
    """
    customers = customer_profiles.copy()

    def get_list_terminals_within_radius(customer_x, customer_y):
        x_dist = terminal_profiles["x_terminal_id"] - customer_x
        y_dist = terminal_profiles["y_terminal_id"] - customer_y
        dist   = np.sqrt(x_dist**2 + y_dist**2)
        return list(terminal_profiles[dist < radius]["TERMINAL_ID"])

    customers["available_terminals"] = customers.apply(
        lambda row: get_list_terminals_within_radius(
            row["x_customer_id"], row["y_customer_id"]
        ),
        axis=1,
    )
    return customers


def generate_transactions_for_customer(
    customer_profile: pd.Series,
    start_date: datetime,
    nb_days: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate the transaction stream for one customer over nb_days."""
    rng = np.random.RandomState(random_state + customer_profile["CUSTOMER_ID"])

    customer_transactions = []
    current_date = start_date

    for day in range(nb_days):
        nb_tx = rng.poisson(customer_profile["mean_nb_tx_per_day"])

        if nb_tx > 0 and len(customer_profile["available_terminals"]) > 0:
            tx_amounts = rng.normal(
                customer_profile["mean_amount"],
                customer_profile["std_amount"],
                nb_tx,
            )
            tx_amounts = np.abs(tx_amounts)  # amounts must be positive

            tx_times = rng.uniform(0, 86400, nb_tx)  # seconds in a day

            for tx_idx in range(nb_tx):
                tx_time = current_date + timedelta(seconds=float(tx_times[tx_idx]))
                terminal_id = rng.choice(customer_profile["available_terminals"])

                customer_transactions.append([
                    tx_time,
                    customer_profile["CUSTOMER_ID"],
                    terminal_id,
                    round(float(tx_amounts[tx_idx]), 2),
                ])

        current_date += timedelta(days=1)

    if not customer_transactions:
        return pd.DataFrame(columns=["TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"])

    return pd.DataFrame(
        customer_transactions,
        columns=["TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"],
    )


def add_fraud_labels(
    transactions: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add fraud labels using three fraud scenarios from the FDH Handbook:

    Scenario 1 — Random fraud (any transaction, any amount)
    Scenario 2 — Compromised terminal (targeted terminals, high amounts)
    Scenario 3 — Compromised customer (targeted customers, small amounts)
    """
    rng = np.random.RandomState(random_state + 999)
    df  = transactions.copy()
    df["TX_FRAUD"] = 0
    df["TX_FRAUD_SCENARIO"] = 0

    # Scenario 1: ~0.1% random fraud baseline
    idx_s1 = rng.choice(len(df), size=max(1, int(0.001 * len(df))), replace=False)
    df.loc[df.index[idx_s1], "TX_FRAUD"] = 1
    df.loc[df.index[idx_s1], "TX_FRAUD_SCENARIO"] = 1

    # Scenario 2: Compromised terminals (rotate every 2 weeks)
    terminals_available = df["TERMINAL_ID"].unique()
    n_compromised = max(1, int(0.003 * len(terminals_available)))

    for day_offset in range(0, df["TX_DATETIME"].dt.dayofyear.max(), 14):
        compromised_terminals = rng.choice(terminals_available, n_compromised, replace=False)
        mask_s2 = (
            df["TERMINAL_ID"].isin(compromised_terminals)
            & (df["TX_AMOUNT"] > 50)
        )
        df.loc[mask_s2, "TX_FRAUD"] = 1
        df.loc[mask_s2, "TX_FRAUD_SCENARIO"] = 2

    # Scenario 3: Compromised customers (small amount, multiple transactions)
    customers_available = df["CUSTOMER_ID"].unique()
    n_compromised_c = max(1, int(0.001 * len(customers_available)))
    compromised_customers = rng.choice(customers_available, n_compromised_c, replace=False)
    mask_s3 = df["CUSTOMER_ID"].isin(compromised_customers) & (df["TX_AMOUNT"] < 20)
    df.loc[mask_s3, "TX_FRAUD"] = 1
    df.loc[mask_s3, "TX_FRAUD_SCENARIO"] = 3

    return df


def generate_dataset(
    n_customers: int = 5000,
    n_terminals: int = 10000,
    nb_days: int = 183,
    start_date: str = "2023-01-01",
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate the full synthetic transaction dataset.

    Args:
        n_customers:  Number of synthetic customer profiles.
        n_terminals:  Number of synthetic terminals (merchants).
        nb_days:      Number of simulation days.
        start_date:   Simulation start date (YYYY-MM-DD).
        random_state: Seed for reproducibility.
        output_path:  If provided, saves the CSV to this path.

    Returns:
        DataFrame with columns:
          TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID,
          TX_AMOUNT, TX_FRAUD, TX_FRAUD_SCENARIO
    """
    log.info(f"Generating dataset: {n_customers} customers, {n_terminals} terminals, {nb_days} days")

    dt_start = datetime.strptime(start_date, "%Y-%m-%d")

    customer_profiles = generate_customer_profiles(n_customers, random_state)
    terminal_profiles = generate_terminal_profiles(n_terminals, random_state)
    customer_profiles = associate_customers_to_terminals(customer_profiles, terminal_profiles)

    # Remove customers with no available terminals
    customer_profiles = customer_profiles[
        customer_profiles["available_terminals"].map(len) > 0
    ]

    log.info(f"  Generating transactions for {len(customer_profiles)} customers ...")
    all_transactions = []
    for _, customer in customer_profiles.iterrows():
        txs = generate_transactions_for_customer(customer, dt_start, nb_days, random_state)
        all_transactions.append(txs)

    df = pd.concat(all_transactions, ignore_index=True)
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)
    df["TRANSACTION_ID"] = range(len(df))

    log.info(f"  Adding fraud labels ...")
    df = add_fraud_labels(df, random_state)

    fraud_rate = df["TX_FRAUD"].mean() * 100
    log.info(
        f"  Done. Transactions: {len(df):,} | "
        f"Fraud: {df['TX_FRAUD'].sum():,} ({fraud_rate:.2f}%)"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        log.info(f"  Saved to {output_path}")

    return df


# ---------------------------------------------------------------------------
# Drift Injection Utilities (for controlled experiments)
# ---------------------------------------------------------------------------

def inject_abrupt_drift(
    df: pd.DataFrame,
    drift_day: int,
    fraud_multiplier: float = 3.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Inject an abrupt drift event at a known day.

    After drift_day:
    - Increases fraud rate sharply
    - Changes fraud pattern: many small transactions instead of large ones

    Args:
        df:               Full transaction DataFrame.
        drift_day:        The day number at which drift begins (0-indexed).
        fraud_multiplier: How many times more fraud after the drift.
        random_state:     Seed for reproducibility.

    Returns:
        Modified DataFrame with TX_DRIFT_INJECTED column marking affected rows.
    """
    rng   = np.random.RandomState(random_state)
    df    = df.copy()
    start = df["TX_DATETIME"].min()

    drift_start_time = start + timedelta(days=drift_day)
    post_drift_mask  = df["TX_DATETIME"] >= drift_start_time

    # Mark drift ground-truth
    df["TX_DRIFT_INJECTED"] = 0
    df.loc[post_drift_mask, "TX_DRIFT_INJECTED"] = 1

    # Increase fraud: label more small transactions as fraud after drift
    post_legit = df[post_drift_mask & (df["TX_FRAUD"] == 0) & (df["TX_AMOUNT"] < 30)]
    n_new_fraud = int(len(post_legit) * 0.05 * fraud_multiplier)
    if n_new_fraud > 0 and len(post_legit) > 0:
        n_new_fraud = min(n_new_fraud, len(post_legit))
        new_fraud_idx = rng.choice(post_legit.index, n_new_fraud, replace=False)
        df.loc[new_fraud_idx, "TX_FRAUD"] = 1
        df.loc[new_fraud_idx, "TX_FRAUD_SCENARIO"] = 4  # new scenario

    log.info(
        f"Abrupt drift injected at day {drift_day}: "
        f"{n_new_fraud:,} new fraud transactions added"
    )
    return df


def inject_gradual_drift(
    df: pd.DataFrame,
    drift_start_day: int,
    drift_end_day: int,
    max_fraud_multiplier: float = 2.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Inject a gradual drift event — fraud rate increases linearly between
    drift_start_day and drift_end_day.
    """
    rng   = np.random.RandomState(random_state)
    df    = df.copy()
    start = df["TX_DATETIME"].min()

    drift_start_time = start + timedelta(days=drift_start_day)
    drift_end_time   = start + timedelta(days=drift_end_day)
    total_drift_seconds = (drift_end_time - drift_start_time).total_seconds()

    df["TX_DRIFT_INJECTED"] = 0
    drift_mask = (df["TX_DATETIME"] >= drift_start_time) & (df["TX_DATETIME"] <= drift_end_time)
    df.loc[drift_mask, "TX_DRIFT_INJECTED"] = 1

    n_total_added = 0
    for _, day_chunk in df[drift_mask & (df["TX_FRAUD"] == 0)].groupby(
        df["TX_DATETIME"].dt.date
    ):
        progress = (
            (day_chunk["TX_DATETIME"].iloc[0] - drift_start_time).total_seconds()
            / max(total_drift_seconds, 1)
        )
        fraud_rate_increase = progress * (max_fraud_multiplier - 1.0) * 0.02
        n_new = int(len(day_chunk) * fraud_rate_increase)
        if n_new > 0:
            idxs = rng.choice(day_chunk.index, min(n_new, len(day_chunk)), replace=False)
            df.loc[idxs, "TX_FRAUD"] = 1
            df.loc[idxs, "TX_FRAUD_SCENARIO"] = 5
            n_total_added += len(idxs)

    log.info(
        f"Gradual drift injected days {drift_start_day}–{drift_end_day}: "
        f"{n_total_added:,} new fraud transactions"
    )
    return df


if __name__ == "__main__":
    # Quick smoke test
    cfg = load_config()
    sim = cfg["simulator"]
    generate_dataset(
        n_customers  = sim["n_customers"],
        n_terminals  = sim["n_terminals"],
        nb_days      = sim["nb_days"],
        start_date   = sim["start_date"],
        random_state = sim["random_state"],
        output_path  = "data/raw/transactions.csv",
    )

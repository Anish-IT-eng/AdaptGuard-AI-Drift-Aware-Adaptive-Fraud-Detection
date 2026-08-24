"""
AdaptGuard AI — Master Experiment Runner
Orchestrates E1–E6 + Oracle experiments and logs all results to MLflow.

Experiments:
  E1 — Stable environment (no drift)
  E2 — Abrupt drift (day 90)
  E3 — Gradual drift (days 60–90)
  E4 — Recurring drift (day 45 + day 90, two separate events)
  E5 — Delayed labels comparison (0 / 1 / 3 / 7 day delays)
  E6 — Safety experiment (validation gate + rollback effectiveness)
  Oracle — Zero-delay upper bound reference

Usage:
  python experiments/run_experiments.py --experiment all
  python experiments/run_experiments.py --experiment e1_stable
  python experiments/run_experiments.py --experiment e4_recurring
  python experiments/run_experiments.py --experiment e5_delayed
  python experiments/run_experiments.py --experiment e6_safety
  python experiments/run_experiments.py --experiment oracle
  python experiments/run_experiments.py --experiment ablation
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

# MLflow (local file tracking — no server required)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.simulator import generate_dataset, inject_abrupt_drift, inject_gradual_drift
from src.data.validator import DataValidator
from src.features.engineering import build_features, get_feature_columns
from src.models.baseline import StaticModel, PeriodicRetrainingModel, build_xgboost
from src.models.online import AlwaysOnlineModel
from src.models.registry import ModelRegistry
from src.evaluation.metrics import evaluate_predictions, compute_business_cost_all_ratios
from src.evaluation.cost import compare_model_costs, compute_business_cost_all_ratios as cost_sensitivity
from src.evaluation.statistical import (
    bootstrap_ci, compare_models, format_results_table, aggregate_rolling_prauc
)
from src.evaluation.ablation import ABLATION_CONFIGS, format_ablation_table, initialize_empty_ablation_table
from sklearn.metrics import average_precision_score

# Streaming multi-model comparison infrastructure
from experiments.streaming_runner import (
    run_streaming_comparison,
    compute_adaptation_gain,
    ExperimentComparison,
)

log = get_logger("experiments.runner")
cfg = load_config()


# ============================================================
# MLflow helpers
# ============================================================

def _mlflow_run(experiment_name: str):
    """Context manager: start an MLflow run if MLflow is available."""
    if not MLFLOW_AVAILABLE:
        from contextlib import contextmanager

        @contextmanager
        def noop():
            yield None

        return noop()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    return mlflow.start_run(run_name=experiment_name)


def _log_params(params: dict) -> None:
    if MLFLOW_AVAILABLE:
        mlflow.log_params({k: str(v) for k, v in params.items()})


def _log_metrics(metrics: dict) -> None:
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        )


def _log_artifact(path: str) -> None:
    if MLFLOW_AVAILABLE and Path(path).exists():
        mlflow.log_artifact(path)


# ============================================================
# Data preparation
# ============================================================

def prepare_data(inject_drift: str = "none", drift_day: int = 90, drift_day_2: int = None):
    """
    Generate and validate the base dataset.

    Args:
        inject_drift:  "none" | "abrupt" | "gradual" | "recurring"
        drift_day:     First (or only) drift injection day.
        drift_day_2:   Second drift day for recurring experiments.

    Returns:
        (train_df, eval_df, feature_cols)
    """
    sim = cfg["simulator"]
    raw_path = "data/raw/transactions.csv"

    # Generate or load
    if not Path(raw_path).exists():
        log.info("Generating simulator dataset ...")
        df = generate_dataset(
            n_customers  = sim["n_customers"],
            n_terminals  = sim["n_terminals"],
            nb_days      = sim["nb_days"],
            start_date   = sim["start_date"],
            random_state = sim["random_state"],
            output_path  = raw_path,
        )
    else:
        log.info(f"Loading existing dataset from {raw_path} ...")
        df = pd.read_csv(raw_path, parse_dates=["TX_DATETIME"])

    # Inject drift for controlled experiments
    if inject_drift == "abrupt":
        df = inject_abrupt_drift(df, drift_day=drift_day, fraud_multiplier=3.0)
    elif inject_drift == "gradual":
        df = inject_gradual_drift(df, drift_start_day=drift_day, drift_end_day=drift_day + 30)
    elif inject_drift == "recurring":
        # E4: Two separate abrupt drift events
        df = inject_abrupt_drift(df, drift_day=drift_day, fraud_multiplier=3.0)
        if drift_day_2:
            df = inject_abrupt_drift(df, drift_day=drift_day_2, fraud_multiplier=2.5)

    # Validate
    validator = DataValidator()
    df, report = validator.validate(df)
    validator.print_report(report)

    # Feature engineering
    log.info("Building features ...")
    df = build_features(df, use_fast_customer_features=True)

    # Save processed
    processed_path = f"data/processed/transactions_{inject_drift}.parquet"
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    log.info(f"Processed dataset saved: {processed_path}")

    feature_cols = get_feature_columns(df)
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]} ...")

    # Chronological split (NO shuffling)
    train_days = cfg["splits"]["train_days"]
    start_date = df["TX_DATETIME"].min()
    split_date = start_date + timedelta(days=train_days)

    train_df = df[df["TX_DATETIME"] < split_date].copy()
    eval_df  = df[df["TX_DATETIME"] >= split_date].copy()

    log.info(
        f"Split: train={len(train_df):,} ({train_days}d) | "
        f"eval={len(eval_df):,}"
    )

    return train_df, eval_df, feature_cols


# ============================================================
# Experiment helpers
# ============================================================

def train_static_xgboost(train_df, feature_cols):
    """Train the static XGBoost baseline."""
    model = StaticModel(build_xgboost(cfg), name="static_xgboost")
    X_tr  = train_df[feature_cols]
    y_tr  = train_df["TX_FRAUD"]

    split = int(0.8 * len(X_tr))
    X_t, X_v = X_tr.iloc[:split], X_tr.iloc[split:]
    y_t, y_v = y_tr.iloc[:split], y_tr.iloc[split:]

    model.model.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)],
        verbose=50,
    )
    model.trained = True
    log.info("Static XGBoost trained.")
    return model


def evaluate_static(model, eval_df, feature_cols, decision_threshold=0.5):
    """Evaluate a static model on the full eval set."""
    X_eval = eval_df[feature_cols]
    y_eval = eval_df["TX_FRAUD"]

    y_proba = model.predict_proba(X_eval)
    y_pred  = (y_proba >= decision_threshold).astype(int)

    metrics = evaluate_predictions(y_eval.values, y_pred, y_proba)
    costs   = compute_business_cost_all_ratios(y_eval.values, y_pred)

    return metrics, costs, y_proba


def log_results(experiment_name, model_name, metrics, costs=None):
    """Log experiment results: print + save to JSON."""
    result = {
        "experiment": experiment_name,
        "model":      model_name,
        "metrics":    metrics,
        "costs":      costs or [],
    }
    out_path = f"results/{experiment_name}_{model_name}.json"
    Path("results").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info(
        f"[{experiment_name}] {model_name}: "
        f"PR-AUC={metrics.get('pr_auc', 'TBD'):.4f} | "
        f"Recall={metrics.get('recall', 'TBD'):.4f} | "
        f"FPR={metrics.get('fpr', 'TBD'):.4f}"
    )
    return result


# ============================================================
# Experiment E1 — Stable Environment
# ============================================================

def run_e1_stable():
    """
    E1: Stable environment — no drift injected.

    Research question: Does AdaptGuard AI avoid unnecessary updates
    in a stable environment?

    Expected: All models similar. AdaptGuard AI should NOT trigger
    adaptation frequently. Results TBD.
    """
    log.info("=" * 60)
    log.info("EXPERIMENT E1: STABLE ENVIRONMENT")
    log.info("=" * 60)

    with _mlflow_run("E1_stable"):
        delay = cfg["delayed_labels"]["default_delay_days"]
        _log_params({
            "experiment":    "E1_stable",
            "inject_drift":  "none",
            "label_delay":   delay,
            "n_customers":   cfg["simulator"]["n_customers"],
            "nb_days":       cfg["simulator"]["nb_days"],
            "train_days":    cfg["splits"]["train_days"],
        })

        train_df, eval_df, feature_cols = prepare_data(inject_drift="none")

        comparison = run_streaming_comparison(
            experiment_name  = "e1_stable",
            train_df         = train_df,
            eval_df          = eval_df,
            feature_cols     = feature_cols,
            drift_scenario   = "none",
            label_delay_days = delay,
            cfg              = cfg,
        )

        # Log all model metrics to MLflow
        for model_name, result in comparison.results.items():
            _log_metrics({f"{model_name}_{k}": v
                          for k, v in result.final_metrics.items()})
            _log_metrics({f"{model_name}_adaptations": result.adaptation_events})

        _log_artifact("results/e1_stable_comparison.json")
        _log_artifact("results/e1_stable_summary.csv")

        log.info("E1 complete.")
    return comparison


# ============================================================
# Experiment E2 — Abrupt Drift
# ============================================================

def run_e2_abrupt():
    """
    E2: Abrupt drift at day 90.

    Research question: How quickly can different systems detect
    and respond to changing fraud behavior?

    Results: TBD
    """
    log.info("=" * 60)
    log.info("EXPERIMENT E2: ABRUPT DRIFT (day 90)")
    log.info("=" * 60)

    with _mlflow_run("E2_abrupt"):
        delay = cfg["delayed_labels"]["default_delay_days"]
        _log_params({
            "experiment":       "E2_abrupt",
            "inject_drift":     "abrupt",
            "drift_day":        90,
            "fraud_multiplier": 3.0,
            "label_delay":      delay,
        })

        train_df, eval_df, feature_cols = prepare_data(inject_drift="abrupt", drift_day=90)

        comparison = run_streaming_comparison(
            experiment_name  = "e2_abrupt",
            train_df         = train_df,
            eval_df          = eval_df,
            feature_cols     = feature_cols,
            drift_scenario   = "abrupt",
            label_delay_days = delay,
            cfg              = cfg,
        )

        for model_name, result in comparison.results.items():
            _log_metrics({f"{model_name}_{k}": v
                          for k, v in result.final_metrics.items()})

        # Adaptation gain: how much does AdaptGuard AI beat static?
        if "static_xgboost" in comparison.results and "adaptguard_ai" in comparison.results:
            gain = compute_adaptation_gain(
                comparison.results["static_xgboost"],
                comparison.results["adaptguard_ai"],
            )
            _log_metrics({"adaptation_gain_prauc": gain["adaptation_gain_prauc"]})
            log.info(f"  Adaptation Gain (PR-AUC): {gain['adaptation_gain_prauc']:.4f} (TBD)")

        _log_artifact("results/e2_abrupt_comparison.json")
        log.info("E2 complete.")
    return comparison


# ============================================================
# Experiment E3 — Gradual Drift
# ============================================================

def run_e3_gradual():
    """
    E3: Gradual drift days 60–90.

    Research question: Can the adaptive system respond to slowly
    evolving behavior?

    Results: TBD
    """
    log.info("=" * 60)
    log.info("EXPERIMENT E3: GRADUAL DRIFT (days 60–90)")
    log.info("=" * 60)

    with _mlflow_run("E3_gradual"):
        delay = cfg["delayed_labels"]["default_delay_days"]
        _log_params({
            "experiment":      "E3_gradual",
            "inject_drift":    "gradual",
            "drift_start_day": 60,
            "drift_end_day":   90,
            "label_delay":     delay,
        })

        train_df, eval_df, feature_cols = prepare_data(inject_drift="gradual", drift_day=60)

        comparison = run_streaming_comparison(
            experiment_name  = "e3_gradual",
            train_df         = train_df,
            eval_df          = eval_df,
            feature_cols     = feature_cols,
            drift_scenario   = "gradual",
            label_delay_days = delay,
            cfg              = cfg,
        )

        for model_name, result in comparison.results.items():
            _log_metrics({f"{model_name}_{k}": v
                          for k, v in result.final_metrics.items()})

        _log_artifact("results/e3_gradual_comparison.json")
        log.info("E3 complete.")
    return comparison


# ============================================================
# Experiment E4 — Recurring Drift
# ============================================================

def run_e4_recurring():
    """
    E4: Recurring drift — two separate abrupt drift events.
      Event 1: day 45  (fraud_multiplier=3.0)
      Event 2: day 90  (fraud_multiplier=2.5)

    Research question: Can AdaptGuard AI repeatedly recover from
    multiple successive drift events without human intervention?

    Results: TBD
    """
    log.info("=" * 60)
    log.info("EXPERIMENT E4: RECURRING DRIFT (day 45 + day 90)")
    log.info("=" * 60)

    with _mlflow_run("E4_recurring"):
        delay = cfg["delayed_labels"]["default_delay_days"]
        _log_params({
            "experiment":        "E4_recurring",
            "inject_drift":      "recurring",
            "drift_day_1":       45,
            "drift_day_2":       90,
            "fraud_multiplier_1": 3.0,
            "fraud_multiplier_2": 2.5,
            "label_delay":       delay,
        })

        train_df, eval_df, feature_cols = prepare_data(
            inject_drift = "recurring",
            drift_day    = 45,
            drift_day_2  = 90,
        )

        comparison = run_streaming_comparison(
            experiment_name  = "e4_recurring",
            train_df         = train_df,
            eval_df          = eval_df,
            feature_cols     = feature_cols,
            drift_scenario   = "recurring",
            label_delay_days = delay,
            cfg              = cfg,
        )

        for model_name, result in comparison.results.items():
            _log_metrics({f"{model_name}_{k}": v
                          for k, v in result.final_metrics.items()})
            _log_metrics({f"{model_name}_adaptations": result.adaptation_events})

        _log_artifact("results/e4_recurring_comparison.json")
        log.info("E4 complete. Key metric: PR-AUC degradation at BOTH drift events?")
    return comparison


# ============================================================
# Experiment E5 — Delayed Labels Study
# ============================================================

def run_e5_delayed_labels():
    """
    E5: Delayed label comparison — 0 / 1 / 3 / 7 day delays.

    Research question: How does label delay affect adaptive system
    performance? Quantifies the "cost of delay."

    Cost of delay = Oracle PR-AUC − AdaptGuard AI (n-day) PR-AUC

    Results: TBD
    """
    log.info("=" * 60)
    log.info("EXPERIMENT E5: DELAYED LABELS (0 / 1 / 3 / 7 days)")
    log.info("=" * 60)

    delay_days_list = cfg["delayed_labels"]["delay_days"]  # [0, 1, 3, 7]
    comparisons_by_delay = {}

    with _mlflow_run("E5_delayed_labels"):
        _log_params({
            "experiment":        "E5_delayed_labels",
            "inject_drift":      "abrupt",
            "drift_day":         90,
            "delay_days_tested": str(delay_days_list),
        })

        # Use abrupt drift — delay cost is most visible under urgency
        train_df, eval_df, feature_cols = prepare_data(inject_drift="abrupt", drift_day=90)

        for delay in delay_days_list:
            log.info(f"  --- E5: label_delay={delay}d ---")

            comparison = run_streaming_comparison(
                experiment_name  = f"e5_delay_{delay}d",
                train_df         = train_df,
                eval_df          = eval_df,
                feature_cols     = feature_cols,
                drift_scenario   = "abrupt",
                label_delay_days = delay,
                cfg              = cfg,
                # For E5, focus on static vs adaptguard (cost of delay comparison)
                run_models       = ["static", "adaptguard"],
            )
            comparisons_by_delay[delay] = comparison

            for model_name, result in comparison.results.items():
                _log_metrics({f"delay{delay}d_{model_name}_{k}": v
                              for k, v in result.final_metrics.items()})

        # Build cost-of-delay table
        _build_delay_cost_table(comparisons_by_delay)
        _log_artifact("results/e5_delay_cost_table.csv")
        log.info("E5 complete.")

    return comparisons_by_delay


def _build_delay_cost_table(comparisons_by_delay: dict) -> pd.DataFrame:
    """Build the delay × model × PR-AUC table for E5 reporting."""
    rows = []
    for delay, comp in comparisons_by_delay.items():
        for model_name, result in comp.results.items():
            rows.append({
                "Label Delay (days)": delay,
                "Model":             model_name,
                "PR-AUC":            round(result.final_metrics.get("pr_auc", 0.0), 4),
                "Recall":            round(result.final_metrics.get("recall", 0.0), 4),
            })
    df = pd.DataFrame(rows)
    out_path = "results/e5_delay_cost_table.csv"
    Path("results").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"E5 delay-cost table saved: {out_path}")
    log.info("\n" + df.to_string(index=False))
    return df


# ============================================================
# Experiment E6 — Safety Experiment
# ============================================================

def run_e6_safety():
    """
    E6: Validation gate + rollback effectiveness.

    Research questions:
      RQ6: Can the validation gate prevent harmful model promotions?
      RQ7: Can rollback recover performance after a bad promotion?

    Primary safety metrics: rejection_count, rollback_count, adaptation_events.
    Results: TBD
    """
    log.info("=" * 60)
    log.info("EXPERIMENT E6: SAFETY (Validation Gate + Rollback)")
    log.info("=" * 60)

    with _mlflow_run("E6_safety"):
        delay = cfg["delayed_labels"]["default_delay_days"]
        _log_params({
            "experiment":            "E6_safety",
            "inject_drift":          "abrupt",
            "drift_day":             90,
            "gate_prauc_improvement": cfg["adaptation"]["gate"]["prauc_improvement"],
            "gate_recall_floor":     cfg["adaptation"]["gate"]["recall_floor"],
            "gate_fpr_ceiling":      cfg["adaptation"]["gate"]["fpr_ceiling"],
            "label_delay_days":      delay,
        })

        train_df, eval_df, feature_cols = prepare_data(inject_drift="abrupt", drift_day=90)

        # E6 focuses on AdaptGuard AI safety metrics under adversarial drift
        comparison = run_streaming_comparison(
            experiment_name  = "e6_safety",
            train_df         = train_df,
            eval_df          = eval_df,
            feature_cols     = feature_cols,
            drift_scenario   = "abrupt",
            label_delay_days = delay,
            cfg              = cfg,
            # Focus: static baseline + full AdaptGuard AI with gate + rollback
            run_models       = ["static", "adaptguard"],
        )

        # Extract safety-specific metrics
        ag_result = comparison.results.get("adaptguard_ai")
        safety_summary = {
            "experiment":           "E6_safety",
            "rejection_count":      ag_result.rejection_count   if ag_result else "TBD",
            "rollback_count":       ag_result.rollback_count    if ag_result else "TBD",
            "adaptation_events":    ag_result.adaptation_events if ag_result else "TBD",
            "adaptguard_pr_auc":    ag_result.final_metrics.get("pr_auc", "TBD") if ag_result else "TBD",
            "note": "All values TBD until experiments complete.",
        }

        for model_name, result in comparison.results.items():
            _log_metrics({f"{model_name}_{k}": v for k, v in result.final_metrics.items()})

        if ag_result:
            _log_metrics({
                "rejection_count":   ag_result.rejection_count,
                "rollback_count":    ag_result.rollback_count,
                "adaptation_events": ag_result.adaptation_events,
            })

        out_path = "results/e6_safety_summary.json"
        Path("results").mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            import json as _json
            _json.dump(safety_summary, f, indent=2, default=str)

        _log_artifact("results/e6_safety_comparison.json")
        _log_artifact(out_path)
        log.info(
            f"E6 complete. Rejections={safety_summary['rejection_count']} | "
            f"Rollbacks={safety_summary['rollback_count']} (TBD)"
        )

    return comparison


# ============================================================
# Oracle Experiment — Zero-delay Upper Bound
# ============================================================

def run_oracle():
    """
    Oracle: Upper-bound reference — AdaptGuard AI with 0-day label delay.

    This is the strongest version of the adaptive system: drift is detected
    and adaptation can begin immediately because labels are available at t=0.

    Cost of delay = Oracle PR-AUC − AdaptGuard AI PR-AUC (at delay=3d)
    Results: TBD
    """
    log.info("=" * 60)
    log.info("ORACLE: ZERO-DELAY UPPER BOUND")
    log.info("=" * 60)

    with _mlflow_run("Oracle_zero_delay"):
        _log_params({
            "experiment":      "oracle",
            "label_delay_days": 0,
            "inject_drift":    "abrupt",
            "drift_day":       90,
            "note":            "0-day delay = perfect label availability",
        })

        train_df, eval_df, feature_cols = prepare_data(inject_drift="abrupt", drift_day=90)

        # Oracle: AdaptGuard AI with 0-day delay (upper bound)
        # Also run static as the lower bound for the same scenario
        comparison = run_streaming_comparison(
            experiment_name  = "oracle",
            train_df         = train_df,
            eval_df          = eval_df,
            feature_cols     = feature_cols,
            drift_scenario   = "abrupt",
            label_delay_days = 0,         # <-- 0-day = oracle condition
            cfg              = cfg,
            run_models       = ["static", "adaptguard"],
        )

        for model_name, result in comparison.results.items():
            _log_metrics({f"oracle_{model_name}_{k}": v
                          for k, v in result.final_metrics.items()})

        oracle_result = comparison.results.get("adaptguard_ai")
        oracle_prauc  = oracle_result.final_metrics.get("pr_auc", 0.0) if oracle_result else 0.0

        # Bootstrap CI on oracle result
        if oracle_result and oracle_result.final_metrics:
            _log_metrics({
                "oracle_adaptguard_prauc":        oracle_prauc,
                "oracle_adaptguard_adaptations":  oracle_result.adaptation_events,
            })

        _log_artifact("results/oracle_comparison.json")
        log.info(
            f"Oracle complete. AdaptGuard AI (0-delay) PR-AUC={oracle_prauc:.4f} (TBD)"
        )

    return comparison


# ============================================================
# Ablation Study
# ============================================================

def run_ablation():
    """
    Run all ablation conditions and produce comparison table.
    Results are TBD until experiments are completed.
    """
    log.info("=" * 60)
    log.info("ABLATION STUDY")
    log.info("=" * 60)

    with _mlflow_run("Ablation_study"):
        _log_params({
            "experiment": "ablation",
            "n_conditions": len(ABLATION_CONFIGS),
            "conditions": str(list(ABLATION_CONFIGS.keys())),
        })

        table = initialize_empty_ablation_table()
        table_path = "results/ablation_table.csv"
        Path("results").mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)

        _log_artifact(table_path)
        log.info(f"Ablation table initialized (TBD): {table_path}")
        log.info("\n" + table.to_string())

    return table


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AdaptGuard AI Experiment Runner")
    parser.add_argument(
        "--experiment",
        default = "e1_stable",
        choices = [
            "all", "e1_stable", "e2_abrupt", "e3_gradual",
            "e4_recurring", "e5_delayed", "e6_safety", "oracle", "ablation"
        ],
        help    = "Which experiment to run",
    )
    args = parser.parse_args()

    if args.experiment == "e1_stable" or args.experiment == "all":
        run_e1_stable()
    if args.experiment == "e2_abrupt" or args.experiment == "all":
        run_e2_abrupt()
    if args.experiment == "e3_gradual" or args.experiment == "all":
        run_e3_gradual()
    if args.experiment == "e4_recurring" or args.experiment == "all":
        run_e4_recurring()
    if args.experiment == "e5_delayed" or args.experiment == "all":
        run_e5_delayed_labels()
    if args.experiment == "e6_safety" or args.experiment == "all":
        run_e6_safety()
    if args.experiment == "oracle" or args.experiment == "all":
        run_oracle()
    if args.experiment == "ablation" or args.experiment == "all":
        run_ablation()

    log.info("All requested experiments complete.")
    log.info("REMINDER: All results are TBD until full experimental runs are completed.")


if __name__ == "__main__":
    main()

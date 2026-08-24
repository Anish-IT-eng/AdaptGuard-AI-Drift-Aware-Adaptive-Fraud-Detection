"""
AdaptGuard AI — Streaming Experiment Runner
Builds, runs, and compares all four model configurations
through the prequential (test-then-train) streaming protocol.

Four configurations compared on every experiment:
  1. static_xgboost    — Train once; never adapt
  2. periodic_7d       — Retrain every 7 days on a rolling 60-day window
  3. always_online     — Update on every confirmed label (SGD)
  4. adaptguard_ai     — Drift-aware selective adaptation (the research system)

All models receive identical data streams in identical temporal order.
All predictions are made BEFORE the label is known.
Labels are released after `label_delay_days` (default 3).

IMPORTANT: All result values are TBD until experiments complete.
This module defines the evaluation structure — not assumed numbers.
"""

import numpy as np
import pandas as pd
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Callable

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.models.baseline import StaticModel, PeriodicRetrainingModel, build_xgboost
from src.models.online import AlwaysOnlineModel
from src.models.registry import ModelRegistry
from src.adaptation.controller import AdaptiveController
from src.evaluation.prequential import PrequentialEvaluator, StreamResult
from src.evaluation.metrics import evaluate_predictions, compute_business_cost_all_ratios
from src.evaluation.statistical import bootstrap_ci
from sklearn.metrics import average_precision_score

log = get_logger("experiments.streaming_runner")


# ============================================================
# Result container
# ============================================================

@dataclass
class ExperimentComparison:
    """
    Complete multi-model comparison result for one experiment.
    All metric values are TBD until experiments complete.
    """
    experiment_name:  str
    drift_scenario:   str
    label_delay_days: int
    results:          dict[str, StreamResult] = field(default_factory=dict)
    run_time_seconds: dict[str, float]        = field(default_factory=dict)
    summary_table:    Optional[pd.DataFrame]  = None

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict."""
        out = {
            "experiment_name":  self.experiment_name,
            "drift_scenario":   self.drift_scenario,
            "label_delay_days": self.label_delay_days,
            "run_time_seconds": self.run_time_seconds,
            "models": {},
        }
        for model_name, result in self.results.items():
            out["models"][model_name] = {
                "final_metrics":    result.final_metrics,
                "n_transactions":   result.n_transactions,
                "adaptation_events": result.adaptation_events,
                "rejection_count":  result.rejection_count,
                "rollback_count":   result.rollback_count,
            }
        return out


# ============================================================
# Model builders
# ============================================================

def _build_static_model(train_df: pd.DataFrame, feature_cols: list[str], cfg: dict) -> StaticModel:
    """Build and train static XGBoost on the training window."""
    model = StaticModel(build_xgboost(cfg), name="static_xgboost")
    X_tr = train_df[feature_cols]
    y_tr = train_df["TX_FRAUD"]

    split = int(0.8 * len(X_tr))
    X_t, X_v = X_tr.iloc[:split], X_tr.iloc[split:]
    y_t, y_v = y_tr.iloc[:split], y_tr.iloc[split:]

    model.model.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)],
        verbose=False,
    )
    model.trained = True
    log.info("Static XGBoost trained.")
    return model


def _build_periodic_model(
    train_df:     pd.DataFrame,
    feature_cols: list[str],
    cfg:          dict,
    interval_days: int = 7,
    window_days:   int = 60,
) -> PeriodicRetrainingModel:
    """Build PeriodicRetrainingModel initialized on the training window."""
    from sklearn.ensemble import RandomForestClassifier

    def factory():
        return RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced", n_jobs=-1,
        )

    model = PeriodicRetrainingModel(
        model_factory          = factory,
        retrain_interval_days  = interval_days,
        window_days            = window_days,
        name                   = f"periodic_{interval_days}d",
    )
    model.initial_fit(
        X     = train_df[feature_cols],
        y     = train_df["TX_FRAUD"],
        dates = train_df["TX_DATETIME"],
    )
    log.info(f"PeriodicRetrainingModel (every {interval_days}d) initialized.")
    return model


def _build_online_model(
    train_df:     pd.DataFrame,
    feature_cols: list[str],
) -> AlwaysOnlineModel:
    """Build AlwaysOnlineModel initialized on the training window."""
    model = AlwaysOnlineModel(name="always_online")
    model.initial_fit(
        X = train_df[feature_cols],
        y = train_df["TX_FRAUD"],
    )
    log.info("AlwaysOnlineModel initialized.")
    return model


def _build_adaptive_controller(
    train_df:      pd.DataFrame,
    feature_cols:  list[str],
    cfg:           dict,
    label_delay_days: int = 3,
    ablation_flags:   Optional[dict] = None,
    models_dir:       str = "models/",
) -> tuple[AdaptiveController, ModelRegistry]:
    """
    Build and initialize AdaptiveController with the training window model
    registered as v1 production.
    """
    # Train initial production model
    static = _build_static_model(train_df, feature_cols, cfg)

    # Set up registry
    registry = ModelRegistry(models_dir=models_dir)
    v1 = registry.register(
        model       = static.model,
        name        = "xgboost_production",
        train_start = str(train_df["TX_DATETIME"].min().date()),
        train_end   = str(train_df["TX_DATETIME"].max().date()),
        metrics     = {},
        hyperparams = cfg["models"]["xgboost"],
        status      = "candidate",
    )
    registry.promote(v1)
    log.info(f"Initial production model registered as v{v1}.")

    # XGBoost factory for candidate training
    def xgb_factory():
        from xgboost import XGBClassifier
        p = cfg["models"]["xgboost"]
        return XGBClassifier(
            n_estimators    = p["n_estimators"],
            max_depth       = p["max_depth"],
            learning_rate   = p["learning_rate"],
            subsample       = p["subsample"],
            colsample_bytree = p["colsample_bytree"],
            scale_pos_weight = p["scale_pos_weight"],
            random_state    = p["random_state"],
            eval_metric     = "aucpr",
            verbosity       = 0,
        )

    controller = AdaptiveController(
        production_model  = static.model,
        model_factory     = xgb_factory,
        registry          = registry,
        feature_cols      = feature_cols,
        reference_df      = train_df,
        cfg               = cfg,
        label_delay_days  = label_delay_days,
        ablation_flags    = ablation_flags,
    )
    return controller, registry


# ============================================================
# Per-model streaming evaluators
# ============================================================

def _run_static_prequential(
    static_model: StaticModel,
    eval_df:      pd.DataFrame,
    feature_cols: list[str],
    delay_days:   int,
) -> tuple[StreamResult, float]:
    """Run static model through full prequential evaluator."""
    t0 = time.time()
    evaluator = PrequentialEvaluator(
        model              = static_model,
        model_name         = static_model.name,
        delay_days         = delay_days,
        decision_threshold = 0.5,
    )
    result = evaluator.run(eval_df, feature_cols)
    elapsed = time.time() - t0
    log.info(f"  Static: {elapsed:.1f}s | PR-AUC={result.final_metrics.get('pr_auc', 'TBD'):.4f}")
    return result, elapsed


def _run_periodic_prequential(
    periodic_model: PeriodicRetrainingModel,
    eval_df:        pd.DataFrame,
    feature_cols:   list[str],
    delay_days:     int,
) -> tuple[StreamResult, float]:
    """
    Run PeriodicRetrainingModel through the prequential loop.
    When a confirmed label arrives, the model observes it and may retrain.
    """
    from src.adaptation.delayed_labels import DelayedLabelBuffer
    from src.evaluation.metrics import RollingMetrics

    t0 = time.time()
    label_buffer    = DelayedLabelBuffer(delay_days=delay_days)
    rolling_metrics = RollingMetrics(window_size=1000)
    all_y_true:  list = []
    all_y_pred:  list = []
    all_y_proba: list = []
    retrain_count = 0

    for i, (_, row) in enumerate(eval_df.iterrows()):
        X_row  = pd.DataFrame([row[feature_cols]])
        y_true = int(row["TX_FRAUD"])
        tx_dt  = row["TX_DATETIME"]

        # PREDICT before label
        y_proba = float(periodic_model.predict_proba(X_row)[0])
        y_pred  = int(y_proba >= 0.5)

        # Store in buffer
        label_buffer.store(
            transaction_id = int(row.get("TRANSACTION_ID", i)),
            tx_datetime    = tx_dt,
            y_true         = y_true,
            y_pred         = y_pred,
            y_prob         = y_proba,
            features       = row[feature_cols],
        )

        # Release confirmed labels
        confirmed = label_buffer.release(current_time=tx_dt)
        for pending in confirmed:
            all_y_true.append(pending.y_true)
            all_y_pred.append(pending.y_pred)
            all_y_proba.append(pending.y_prob)
            rolling_metrics.update(pending.y_true, pending.y_pred, pending.y_prob, tx_dt)

            # Observe → may trigger retrain
            did_retrain = periodic_model.observe(
                X_row        = pending.features,
                y            = pending.y_true,
                current_date = pending.tx_datetime,
            )
            if did_retrain:
                retrain_count += 1

        if (i + 1) % 50_000 == 0:
            recent_prauc = (
                rolling_metrics.history[-1]["pr_auc"]
                if rolling_metrics.history else 0.0
            )
            log.info(
                f"  [periodic] {i+1:,} tx | "
                f"Retrains={retrain_count} | "
                f"Rolling PR-AUC={recent_prauc:.4f}"
            )

    final_metrics = evaluate_predictions(
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)
    ) if all_y_true else {}

    elapsed = time.time() - t0
    log.info(
        f"  Periodic ({periodic_model.retrain_interval_days}d): {elapsed:.1f}s | "
        f"PR-AUC={final_metrics.get('pr_auc', 'TBD'):.4f} | "
        f"Retrains={retrain_count}"
    )

    return StreamResult(
        model_name        = periodic_model.name,
        n_transactions    = len(eval_df),
        n_fraud_detected  = final_metrics.get("tp", 0),
        n_fraud_total     = final_metrics.get("n_fraud", 0),
        adaptation_events = retrain_count,
        rejection_count   = 0,
        rollback_count    = 0,
        final_metrics     = final_metrics,
        rolling_history   = rolling_metrics.history.copy(),
        label_delay_days  = delay_days,
    ), elapsed


def _run_online_prequential(
    online_model: AlwaysOnlineModel,
    eval_df:      pd.DataFrame,
    feature_cols: list[str],
    delay_days:   int,
) -> tuple[StreamResult, float]:
    """
    Run AlwaysOnlineModel through the prequential loop.
    Updates on every confirmed label.
    """
    from src.adaptation.delayed_labels import DelayedLabelBuffer
    from src.evaluation.metrics import RollingMetrics

    t0 = time.time()
    label_buffer    = DelayedLabelBuffer(delay_days=delay_days)
    rolling_metrics = RollingMetrics(window_size=1000)
    all_y_true:  list = []
    all_y_pred:  list = []
    all_y_proba: list = []

    for i, (_, row) in enumerate(eval_df.iterrows()):
        X_row  = pd.DataFrame([row[feature_cols]])
        y_true = int(row["TX_FRAUD"])
        tx_dt  = row["TX_DATETIME"]

        # PREDICT before label
        y_proba = float(online_model.predict_proba(X_row)[0])
        y_pred  = int(y_proba >= 0.5)

        # Store in buffer
        label_buffer.store(
            transaction_id = int(row.get("TRANSACTION_ID", i)),
            tx_datetime    = tx_dt,
            y_true         = y_true,
            y_pred         = y_pred,
            y_prob         = y_proba,
            features       = row[feature_cols],
        )

        # Release confirmed labels
        confirmed = label_buffer.release(current_time=tx_dt)
        for pending in confirmed:
            all_y_true.append(pending.y_true)
            all_y_pred.append(pending.y_pred)
            all_y_proba.append(pending.y_prob)
            rolling_metrics.update(pending.y_true, pending.y_pred, pending.y_prob, tx_dt)

            # Update on every confirmed label
            X_conf = pd.DataFrame([pending.features])
            online_model.observe(X_conf, pending.y_true)

        if (i + 1) % 50_000 == 0:
            recent_prauc = (
                rolling_metrics.history[-1]["pr_auc"]
                if rolling_metrics.history else 0.0
            )
            log.info(
                f"  [always_online] {i+1:,} tx | "
                f"Updates={online_model.update_count:,} | "
                f"Rolling PR-AUC={recent_prauc:.4f}"
            )

    final_metrics = evaluate_predictions(
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)
    ) if all_y_true else {}

    elapsed = time.time() - t0
    log.info(
        f"  Always-Online: {elapsed:.1f}s | "
        f"PR-AUC={final_metrics.get('pr_auc', 'TBD'):.4f} | "
        f"Updates={online_model.update_count:,}"
    )

    return StreamResult(
        model_name        = online_model.name,
        n_transactions    = len(eval_df),
        n_fraud_detected  = final_metrics.get("tp", 0),
        n_fraud_total     = final_metrics.get("n_fraud", 0),
        adaptation_events = online_model.update_count,
        rejection_count   = 0,
        rollback_count    = 0,
        final_metrics     = final_metrics,
        rolling_history   = rolling_metrics.history.copy(),
        label_delay_days  = delay_days,
    ), elapsed


def _run_adaptguard_prequential(
    controller:  AdaptiveController,
    registry:    ModelRegistry,
    eval_df:     pd.DataFrame,
    feature_cols: list[str],
) -> tuple[StreamResult, float]:
    """
    Run AdaptGuard AI through the prequential loop via controller.process_transaction().
    The controller handles its own internal delayed-label buffer.
    """
    from src.evaluation.metrics import RollingMetrics

    t0 = time.time()
    rolling_metrics = RollingMetrics(window_size=1000)
    all_y_true:  list = []
    all_y_pred:  list = []
    all_y_proba: list = []

    for i, (_, row) in enumerate(eval_df.iterrows()):
        tx_dt = row["TX_DATETIME"]

        # controller.process_transaction handles predict + buffer + release + drift + adapt
        y_pred, y_prob = controller.process_transaction(row=row, timestamp=tx_dt)

        all_y_true.append(int(row["TX_FRAUD"]))
        all_y_pred.append(y_pred)
        all_y_proba.append(y_prob)
        rolling_metrics.update(int(row["TX_FRAUD"]), y_pred, y_prob, tx_dt)

        if (i + 1) % 50_000 == 0:
            recent_prauc = (
                rolling_metrics.history[-1]["pr_auc"]
                if rolling_metrics.history else 0.0
            )
            log.info(
                f"  [adaptguard] {i+1:,} tx | "
                f"Adaptations={controller.adaptation_count} | "
                f"Rejections={controller.rejection_count} | "
                f"Rolling PR-AUC={recent_prauc:.4f}"
            )

    final_metrics = evaluate_predictions(
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)
    ) if all_y_true else {}

    rollback_count = getattr(registry, "rollback_count", 0)
    elapsed = time.time() - t0
    log.info(
        f"  AdaptGuard AI: {elapsed:.1f}s | "
        f"PR-AUC={final_metrics.get('pr_auc', 'TBD'):.4f} | "
        f"Adaptations={controller.adaptation_count} | "
        f"Rejections={controller.rejection_count} | "
        f"Rollbacks={rollback_count}"
    )

    return StreamResult(
        model_name        = "adaptguard_ai",
        n_transactions    = len(eval_df),
        n_fraud_detected  = final_metrics.get("tp", 0),
        n_fraud_total     = final_metrics.get("n_fraud", 0),
        adaptation_events = controller.adaptation_count,
        rejection_count   = controller.rejection_count,
        rollback_count    = rollback_count,
        final_metrics     = final_metrics,
        rolling_history   = rolling_metrics.history.copy(),
        label_delay_days  = controller.label_buffer.delay_days,
    ), elapsed


# ============================================================
# Main multi-model comparison function
# ============================================================

def run_streaming_comparison(
    experiment_name:   str,
    train_df:          pd.DataFrame,
    eval_df:           pd.DataFrame,
    feature_cols:      list[str],
    drift_scenario:    str = "none",
    label_delay_days:  int = 3,
    cfg:               Optional[dict] = None,
    ablation_flags:    Optional[dict] = None,
    models_dir:        str = "models/",
    results_dir:       str = "results/",
    run_models:        Optional[list[str]] = None,
) -> ExperimentComparison:
    """
    Run the full prequential multi-model comparison for one experiment.

    Trains all four model configurations on `train_df` and evaluates
    them on `eval_df` using the prequential streaming protocol.

    Args:
        experiment_name:   Identifier (e.g., "e1_stable", "e2_abrupt").
        train_df:          Training window (chronologically ordered).
        eval_df:           Evaluation window (chronologically ordered).
        feature_cols:      Feature column names.
        drift_scenario:    "none" | "abrupt" | "gradual" | "recurring".
        label_delay_days:  Label delay for all models (0=oracle, 3=default).
        cfg:               Config dict.
        ablation_flags:    Ablation flags for AdaptGuard AI (None = full system).
        models_dir:        Path for model artifacts.
        results_dir:       Path for JSON results output.
        run_models:        List of model keys to run. None = run all four.
                           Options: ["static", "periodic", "online", "adaptguard"]

    Returns:
        ExperimentComparison with all StreamResults and summary table.
    """
    cfg = cfg or load_config()
    run_models = run_models or ["static", "periodic", "online", "adaptguard"]
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info(f"STREAMING COMPARISON: {experiment_name}")
    log.info(f"  Drift: {drift_scenario} | Delay: {label_delay_days}d | "
             f"Train: {len(train_df):,} | Eval: {len(eval_df):,}")
    log.info(f"  Models: {run_models}")
    log.info("=" * 70)

    comparison = ExperimentComparison(
        experiment_name  = experiment_name,
        drift_scenario   = drift_scenario,
        label_delay_days = label_delay_days,
    )

    # ---- 1. STATIC XGBoost ----
    if "static" in run_models:
        log.info("[1/4] Static XGBoost ...")
        static_model = _build_static_model(train_df, feature_cols, cfg)
        result, elapsed = _run_static_prequential(
            static_model, eval_df, feature_cols, label_delay_days
        )
        comparison.results["static_xgboost"]    = result
        comparison.run_time_seconds["static_xgboost"] = elapsed

    # ---- 2. Periodic Retraining ----
    if "periodic" in run_models:
        log.info("[2/4] Periodic Retraining (7-day) ...")
        periodic_model = _build_periodic_model(
            train_df, feature_cols, cfg,
            interval_days = cfg.get("baselines", {}).get("periodic_interval_days", 7),
        )
        result, elapsed = _run_periodic_prequential(
            periodic_model, eval_df, feature_cols, label_delay_days
        )
        comparison.results["periodic_7d"]    = result
        comparison.run_time_seconds["periodic_7d"] = elapsed

    # ---- 3. Always-Online ----
    if "online" in run_models:
        log.info("[3/4] Always-Online (SGD) ...")
        online_model = _build_online_model(train_df, feature_cols)
        result, elapsed = _run_online_prequential(
            online_model, eval_df, feature_cols, label_delay_days
        )
        comparison.results["always_online"]    = result
        comparison.run_time_seconds["always_online"] = elapsed

    # ---- 4. AdaptGuard AI ----
    if "adaptguard" in run_models:
        log.info("[4/4] AdaptGuard AI (Adaptive Controller) ...")
        run_id = f"{experiment_name}_adaptguard"
        m_dir  = str(Path(models_dir) / run_id)
        controller, registry = _build_adaptive_controller(
            train_df, feature_cols, cfg,
            label_delay_days = label_delay_days,
            ablation_flags   = ablation_flags,
            models_dir       = m_dir,
        )
        result, elapsed = _run_adaptguard_prequential(
            controller, registry, eval_df, feature_cols
        )
        comparison.results["adaptguard_ai"]    = result
        comparison.run_time_seconds["adaptguard_ai"] = elapsed

    # ---- Summary table ----
    comparison.summary_table = _build_summary_table(comparison)

    # ---- Save results ----
    out_path = Path(results_dir) / f"{experiment_name}_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2, default=str)

    csv_path = Path(results_dir) / f"{experiment_name}_summary.csv"
    if comparison.summary_table is not None:
        comparison.summary_table.to_csv(csv_path, index=False)

    log.info(f"Results saved: {out_path}")
    log.info(f"Summary:\n{comparison.summary_table.to_string(index=False)}"
             if comparison.summary_table is not None else "")

    return comparison


# ============================================================
# Summary table builder
# ============================================================

def _build_summary_table(comparison: ExperimentComparison) -> pd.DataFrame:
    """Build the research comparison table from all model results."""
    rows = []
    for model_name, result in comparison.results.items():
        m = result.final_metrics
        rows.append({
            "Experiment":    comparison.experiment_name,
            "Drift":         comparison.drift_scenario,
            "Model":         model_name,
            "Delay(d)":      comparison.label_delay_days,
            "PR-AUC":        round(m.get("pr_auc",    0.0), 4),
            "Recall":        round(m.get("recall",    0.0), 4),
            "Precision":     round(m.get("precision", 0.0), 4),
            "FPR":           round(m.get("fpr",       0.0), 4),
            "F2":            round(m.get("f2",        0.0), 4),
            "Adaptations":   result.adaptation_events,
            "Rejections":    result.rejection_count,
            "Rollbacks":     result.rollback_count,
            "Runtime(s)":    round(comparison.run_time_seconds.get(model_name, 0.0), 1),
        })
    return pd.DataFrame(rows)


# ============================================================
# Adaptation gain computation
# ============================================================

def compute_adaptation_gain(
    static_result:    StreamResult,
    adaptive_result:  StreamResult,
) -> dict:
    """
    Compute adaptation gain: how much does AdaptGuard AI improve over static?

    Adaptation Gain = Adaptive PR-AUC − Static PR-AUC

    Results TBD until experiments complete.
    """
    static_prauc   = static_result.final_metrics.get("pr_auc", 0.0)
    adaptive_prauc = adaptive_result.final_metrics.get("pr_auc", 0.0)

    return {
        "adaptation_gain_prauc": round(adaptive_prauc - static_prauc, 4),
        "static_prauc":          round(static_prauc, 4),
        "adaptive_prauc":        round(adaptive_prauc, 4),
        "adaptation_events":     adaptive_result.adaptation_events,
        "rejection_count":       adaptive_result.rejection_count,
        "rollback_count":        adaptive_result.rollback_count,
        "note": "TBD until full experiments complete",
    }

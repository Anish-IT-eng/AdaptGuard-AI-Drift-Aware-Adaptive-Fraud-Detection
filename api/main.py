"""
AdaptGuard AI — FastAPI Backend
Real-time fraud prediction, drift status, model info, and adaptation endpoints.

IMPORTANT: /adapt and /rollback are exposed for research/demo purposes only.
In real production systems these would not be unrestricted public endpoints.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("api.main")

# ============================================================
# Application State
# ============================================================

app_state: dict[str, Any] = {
    "controller":   None,
    "registry":     None,
    "feature_cols": [],
    "n_predictions": 0,
    "start_time":    datetime.utcnow().isoformat(),
    "last_drift_result": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    log.info("AdaptGuard AI API starting up ...")
    try:
        from src.models.registry import ModelRegistry
        registry = ModelRegistry(models_dir="models")
        app_state["registry"] = registry
        log.info("Model registry loaded.")
    except Exception as e:
        log.warning(f"Registry not initialized: {e}")
    yield
    log.info("AdaptGuard AI API shutting down.")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title       = "AdaptGuard AI",
    description = (
        "Drift-Aware Adaptive Fraud Detection API.\n\n"
        "Research prototype — results are TBD until experiments complete."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ============================================================
# Schemas
# ============================================================

class TransactionRequest(BaseModel):
    transaction_id: Optional[int]   = Field(None,  description="Unique transaction identifier")
    tx_datetime:    str              = Field(...,   description="Transaction timestamp (ISO format)")
    customer_id:    str              = Field(...,   description="Customer identifier")
    terminal_id:    str              = Field(...,   description="Terminal/merchant identifier")
    tx_amount:      float            = Field(...,   ge=0, description="Transaction amount")
    payment_channel: Optional[str]  = Field("card", description="online | card | mobile")
    features:       Optional[dict]  = Field(None,  description="Pre-computed engineered features")


class PredictionResponse(BaseModel):
    transaction_id:   Optional[int]
    fraud_probability: float
    decision:          str           # "FRAUD" | "LEGITIMATE"
    model_version:     int
    inference_latency_ms: float
    timestamp:         str


class DriftStatusResponse(BaseModel):
    drift_detected:  bool
    severity_level:  str
    severity_score:  float
    adwin_signal:    bool
    max_psi:         float
    error_rate:      float
    explanation:     str
    timestamp:       str


class ModelInfoResponse(BaseModel):
    production_version: Optional[int]
    model_name:         str
    status:             str
    train_start:        str
    train_end:          str
    metrics:            dict
    adaptation_count:   int
    rejection_count:    int
    rollback_count:     int


class MetricsResponse(BaseModel):
    pr_auc:    float
    recall:    float
    precision: float
    f1:        float
    fpr:       float
    n_samples: int
    timestamp: str


class AdaptRequest(BaseModel):
    reason: Optional[str] = "manual_trigger"


class RollbackRequest(BaseModel):
    reason: Optional[str] = "manual_rollback"


# ============================================================
# Endpoints
# ============================================================

@app.get("/", tags=["Health"])
async def root():
    return {
        "system":  "AdaptGuard AI",
        "version": "1.0.0",
        "status":  "running",
        "uptime":  app_state["start_time"],
        "note":    "Research prototype — results are TBD until experiments complete.",
    }


@app.get("/health", tags=["Health"])
async def health():
    registry = app_state.get("registry")
    return {
        "status":           "ok",
        "production_model": registry.production_version if registry else None,
        "n_predictions":    app_state["n_predictions"],
        "timestamp":        datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: TransactionRequest):
    """
    Generate a fraud prediction for a single transaction.

    Input: Transaction features
    Output: Fraud probability, binary decision, model version, latency
    """
    registry = app_state.get("registry")
    if registry is None or registry.production_version is None:
        raise HTTPException(status_code=503, detail="No production model available.")

    t0 = time.time()

    try:
        prod_model, record = registry.get_production_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    # Build feature vector from request
    # In production: apply full feature pipeline
    # For demo: use provided features dict or basic features
    if request.features:
        X = pd.DataFrame([request.features])
    else:
        # Minimal feature set when full pipeline isn't available
        X = pd.DataFrame([{
            "TX_AMOUNT": request.tx_amount,
            "TX_HOUR":   pd.to_datetime(request.tx_datetime).hour,
            "TX_DAY_OF_WEEK": pd.to_datetime(request.tx_datetime).dayofweek,
            "TX_IS_WEEKEND": int(pd.to_datetime(request.tx_datetime).dayofweek >= 5),
        }])

    feature_cols = app_state.get("feature_cols", list(X.columns))
    available_cols = [c for c in feature_cols if c in X.columns]
    if not available_cols:
        available_cols = list(X.columns)

    try:
        proba = prod_model.predict_proba(X[available_cols])[:, 1]
        prob  = float(proba[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    latency_ms = (time.time() - t0) * 1000
    app_state["n_predictions"] += 1

    return PredictionResponse(
        transaction_id      = request.transaction_id,
        fraud_probability   = round(prob, 4),
        decision            = "FRAUD" if prob >= 0.5 else "LEGITIMATE",
        model_version       = record.version,
        inference_latency_ms = round(latency_ms, 3),
        timestamp           = datetime.utcnow().isoformat(),
    )


@app.get("/drift", response_model=DriftStatusResponse, tags=["Monitoring"])
async def get_drift_status():
    """Get current drift monitoring status from both channels."""
    controller = app_state.get("controller")

    if controller is None:
        # Return demo status when controller not initialized
        return DriftStatusResponse(
            drift_detected = False,
            severity_level = "NONE",
            severity_score = 0.0,
            adwin_signal   = False,
            max_psi        = 0.0,
            error_rate     = 0.0,
            explanation    = "Controller not initialized — run experiments first.",
            timestamp      = datetime.utcnow().isoformat(),
        )

    status = controller.get_status()
    return DriftStatusResponse(
        drift_detected = status["severity_level"] not in ("NONE", "LOW"),
        severity_level = status["severity_level"],
        severity_score = round(status["severity_score"], 4),
        adwin_signal   = status["adwin_signal"],
        max_psi        = round(status["max_psi"], 4),
        error_rate     = round(status["error_rate"], 4),
        explanation    = f"Production v{status['production_version']} | {status}",
        timestamp      = datetime.utcnow().isoformat(),
    )


@app.get("/model", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get current production model metadata."""
    registry = app_state.get("registry")
    if registry is None or registry.production_version is None:
        raise HTTPException(status_code=503, detail="No production model.")

    _, record = registry.get_production_model()
    controller = app_state.get("controller")

    return ModelInfoResponse(
        production_version = record.version,
        model_name         = record.name,
        status             = record.status,
        train_start        = record.train_start,
        train_end          = record.train_end,
        metrics            = record.metrics,
        adaptation_count   = controller.adaptation_count if controller else 0,
        rejection_count    = controller.rejection_count  if controller else 0,
        rollback_count     = registry.rollback_count,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get current model performance metrics."""
    controller = app_state.get("controller")

    if controller is None:
        return MetricsResponse(
            pr_auc=0.0, recall=0.0, precision=0.0,
            f1=0.0, fpr=0.0, n_samples=0,
            timestamp=datetime.utcnow().isoformat(),
        )

    history = controller._rolling.history
    if history:
        latest = history[-1]
        return MetricsResponse(
            pr_auc    = round(latest.get("pr_auc", 0.0), 4),
            recall    = round(latest.get("recall", 0.0), 4),
            precision = round(latest.get("precision", 0.0), 4),
            f1        = round(latest.get("f1", 0.0), 4),
            fpr       = round(latest.get("fpr", 0.0), 4),
            n_samples = latest.get("n_total", 0),
            timestamp = datetime.utcnow().isoformat(),
        )

    return MetricsResponse(
        pr_auc=0.0, recall=0.0, precision=0.0,
        f1=0.0, fpr=0.0, n_samples=0,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/adapt", tags=["Control"])
async def trigger_adaptation(request: AdaptRequest):
    """
    Manually trigger an adaptation cycle.

    RESEARCH/DEMO USE ONLY. Not for unrestricted production access.
    """
    controller = app_state.get("controller")
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized.")

    log.info(f"Manual adaptation triggered: reason={request.reason}")
    return {
        "triggered": True,
        "reason":    request.reason,
        "timestamp": datetime.utcnow().isoformat(),
        "note":      "Adaptation cycle queued. Check /model for updated version.",
    }


@app.post("/rollback", tags=["Control"])
async def trigger_rollback(request: RollbackRequest):
    """
    Manually trigger a model rollback.

    RESEARCH/DEMO USE ONLY.
    This rolls back from current production to the most recent stable archived model.
    """
    registry = app_state.get("registry")
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized.")

    try:
        restored_version = registry.rollback(reason=request.reason or "manual_rollback")
        return {
            "success":          True,
            "restored_version": restored_version,
            "reason":           request.reason,
            "timestamp":        datetime.utcnow().isoformat(),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/history", tags=["Monitoring"])
async def get_adaptation_history():
    """Get full history of adaptation events."""
    controller = app_state.get("controller")
    if controller is None:
        return {"events": [], "total": 0}

    events = [
        {
            "timestamp":       e.timestamp,
            "action":          e.action,
            "severity_level":  e.severity_level,
            "severity_score":  e.severity_score,
            "production_v":    e.production_v,
            "candidate_v":     e.candidate_v,
            "validation":      e.validation_result,
        }
        for e in controller.adaptation_events
    ]

    return {
        "events":           events,
        "total":            len(events),
        "adaptation_count": controller.adaptation_count,
        "rejection_count":  controller.rejection_count,
    }


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(
        "api.main:app",
        host   = cfg["api"]["host"],
        port   = cfg["api"]["port"],
        reload = cfg["api"]["reload"],
    )

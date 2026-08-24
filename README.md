# AdaptGuard AI
### Drift-Aware Adaptive Fraud Detection Under Evolving Transaction Distributions

> **Research prototype.** All performance numbers are TBD until controlled experiments are completed.

---

## Overview

AdaptGuard AI is a research system that investigates whether a **drift-aware selective adaptation controller** can reduce performance degradation under evolving transaction distributions — compared with static, periodic-retraining, and always-online approaches — while avoiding unnecessary model updates.

The fraud detection setting provides the experimental context. The **adaptive controller** is the research subject.

---

## Core Research Question

> *Can a drift-aware adaptive controller detect and respond to concept drift in streaming fraud data, while avoiding unnecessary updates in stable periods?*

---

## Architecture

```
TRANSACTION (no label yet)
        │
        ├──→ DATA DRIFT MONITOR (PSI / KS / MMD)        ← Label-free, fires immediately
        │
        └──→ FRAUD MODEL → Prediction stored in buffer
                              │
                              ↓ (1–7 days later)
                          LABEL ARRIVES
                              │
                          PERFORMANCE MONITOR (ADWIN / PH) ← Label-required
                              │
                          SEVERITY ESTIMATOR
                              │
                        ┌─────▼────────┐
                        │ ADAPTIVE     │
                        │ CONTROLLER   │
                        └──────┬───────┘
                               │
                    ┌──────────▼───────────┐
                    │  VALIDATION GATE     │
                    │  (Champion-Challenger)│
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │ PROMOTE or REJECT   │
                    │ (never blind swap)  │
                    └─────────────────────┘
                               │ (if promoted, monitor post-deployment)
                    ┌──────────▼──────────┐
                    │ ROLLBACK if needed  │
                    └─────────────────────┘
```

---

## Experiments

| Experiment | Drift Scenario | Research Question |
|---|---|---|
| **E1** — Stable | No drift | Does the system avoid unnecessary updates? |
| **E2** — Abrupt | Sudden shift at day 90 | How quickly is drift detected and recovered? |
| **E3** — Gradual | Slow shift days 60–90 | Can slow-moving drift be detected? |
| **E4** — Recurring | Two events: days 45 + 90 | Can the system adapt multiple times? |
| **E5** — Delayed Labels | 0 / 1 / 3 / 7 day delays | What is the cost of label delay? |
| **E6** — Safety | Adversarial conditions | Does the validation gate prevent harmful updates? |
| **Oracle** | Zero-delay upper bound | What is maximum achievable performance? |

---

## Baselines

| Model | Description |
|---|---|
| **Static XGBoost** | Never adapts — lower bound under drift |
| **Periodic Retraining** | Adapts on calendar schedule (1 / 7 / 14 / 30 days) |
| **Always-Online SGD** | Updates on every confirmed label regardless of drift |
| **AdaptGuard AI** | Adapts only when drift evidence warrants it |
| **Oracle** | Zero-delay adaptation — upper bound reference |

---

## Primary Metric

**PR-AUC** (Area under Precision-Recall Curve)  
Required due to severe class imbalance (~0.84% base fraud rate).  
Accuracy is explicitly deprecated as a primary metric.

---

## Project Structure

```
AdaptGuard-AI/
├── src/
│   ├── data/          ← Simulator + validator + loader
│   ├── features/      ← Leak-free feature engineering
│   ├── models/        ← Baselines + online models + registry
│   ├── drift/         ← PSI, KS, MMD, ADWIN, PH, severity, orchestrator
│   ├── adaptation/    ← Controller, candidate trainer, delayed labels
│   └── evaluation/    ← Metrics, cost, prequential, statistical, ablation
├── api/               ← FastAPI REST backend
├── dashboard/         ← Streamlit monitoring dashboard
├── experiments/       ← Experiment runner (E1–E6 + Oracle)
├── tests/             ← Unit tests (pytest)
├── configs/           ← config.yaml with all parameters
├── data/              ← Raw + processed transaction data (generated locally)
├── models/            ← Serialized model artifacts
└── results/           ← Experiment JSON outputs
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data + Run E1 (Stable Baseline)

```bash
python experiments/run_experiments.py --experiment e1_stable
```

### 3. Run All Experiments

```bash
python experiments/run_experiments.py --experiment all
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### 5. Start API

```bash
uvicorn api.main:app --reload
```

### 6. Start Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Synthetic data** | Controlled drift injection with known timestamps for delay measurement |
| **Prequential evaluation** | Test-then-train protocol — required for temporal + delayed-label setup |
| **Dual monitoring channels** | PSI/KS/MMD (label-free) + ADWIN/PH (label-required) — different latencies |
| **Champion-Challenger gate** | Prevents harmful promotions before production exposure |
| **Rejection ≠ Rollback** | Rejection: candidate never deployed. Rollback: deployed model later reverted |
| **No shuffling** | All splits are chronological — shuffling would be temporal leakage |

---

## Dataset

ULB/Worldline Fraud Detection Handbook simulator.  
See [`data/README.md`](data/README.md) for full details.

---

## Results

> ⚠️ All results are **TBD** until controlled experiments are completed.  
> No numbers are assumed in advance.

| Metric | Static XGBoost | Periodic Retrain | Always-Online | AdaptGuard AI |
|---|---|---|---|---|
| PR-AUC (E1 Stable) | TBD | TBD | TBD | TBD |
| PR-AUC (E2 Abrupt) | TBD | TBD | TBD | TBD |
| PR-AUC (E3 Gradual) | TBD | TBD | TBD | TBD |
| Detection Delay | N/A | N/A | TBD | TBD |
| Rejected Candidates | N/A | N/A | N/A | TBD |
| Rollback Count | N/A | N/A | N/A | TBD |

---

## References

- Le Borgne et al. (2022) — *Reproducible Machine Learning for Credit Card Fraud Detection*  
  https://fraud-detection-handbook.github.io/fraud-detection-handbook/
- Bifet & Gavalda (2007) — *Learning from Time-Changing Data with Adaptive Windowing* (ADWIN)
- River ML Library — https://riverml.xyz/
- MLflow — https://mlflow.org/

---

*Version 1.0.0 · Research prototype · Results TBD*

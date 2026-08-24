
# AdaptGuard AI — Understanding Report & Implementation Plan v2

> Based on the refined v2 project specification. This version captures all new distinctions, additions, and corrections from the updated spec.

---

## PART 1 — DEEP UNDERSTANDING REPORT

### 1.1 What This Project Actually Is

AdaptGuard AI is a **research experiment**, not a fraud classifier. The core deliverable is an answer — established through controlled experiments — to a single research question:

> *"Can a drift-aware selective adaptation controller reduce performance degradation under evolving transaction distributions, compared with static, periodic-retraining, and always-online approaches, while avoiding unnecessary updates?"*

The fraud detection setting is the context. The drift-aware adaptation system is the research subject.

---

### 1.2 The Two Types of Drift — Critical Distinction (New in v2)

This is the most architecturally important distinction in the v2 spec. The system must monitor **two fundamentally different things** with **different tools** because they have different label dependencies.

| Drift Type | What Changes | Math | Label Required? | Detection Tools |
|---|---|---|---|---|
| **Data / Covariate Drift** | Input distributions | P(X) changes | ❌ No | PSI, KS test, MMD |
| **Concept / Performance Drift** | Feature-to-fraud relationship | P(Y\|X) changes | ✅ Yes (confirmed labels) | ADWIN, Page-Hinkley |

**Why this matters architecturally:**

- PSI/KS/MMD can fire **immediately** when a transaction arrives, because they compare feature distributions against a reference window — no label needed.
- ADWIN (when used on prediction errors) can only fire **after a label is confirmed** — which in this system takes 1–7 days. Until the label arrives, ADWIN has no error signal to process.

Therefore the system has **two parallel monitoring channels**:

```
TRANSACTION (no label yet)
        │
        ├──→ DATA DRIFT MONITOR (PSI / KS / MMD)
        │    Fires immediately — no label dependency
        │    Detects: feature distribution changes
        │
        └──→ FRAUD MODEL → Prediction stored in buffer
                              │
                              ↓ (1–7 days later)
                          LABEL ARRIVES
                              │
                          PERFORMANCE MONITOR (ADWIN / PH)
                          Fires after label confirmation
                          Detects: model error rate changes
```

This dual-channel architecture is research-verified: PSI/KS for covariate drift (label-free), ADWIN for concept drift (label-dependent). Both signals feed the Adaptive Controller, which uses **both together** to estimate severity.

---

### 1.3 Rejection vs. Rollback — Critical Distinction (New in v2)

These are two completely different events that the spec explicitly separates. Confusing them in the implementation would be a significant error.

| Scenario | Production Model | Candidate | What Happens | Name |
|---|---|---|---|---|
| **Case 1** | v1 running | v2 trained, fails validation | v2 never touches production | **REJECTION** |
| **Case 2** | v2 promoted, then degrades | v1 archived | v1 is restored from rollback store | **ROLLBACK** |

```
CASE 1 — REJECTION (no rollback needed):
v1 Production → candidate v2 → validation fails → v2 discarded → v1 continues
                                                   ↑ No rollback event

CASE 2 — ROLLBACK (v2 already promoted):
v1 → v2 passes validation → v2 promoted → v2 degrades post-deployment → v1 restored
                                                                          ↑ This is rollback
```

Both must be separately tracked in the MLflow experiment log:
- `rejected_candidates_count` — how many never reached production
- `rollback_count` — how many promoted models were later reverted

---

### 1.4 What the Adaptive Controller Actually Does

The controller is the core research component. It does NOT simply update the model on every new transaction (that would be always-online learning). It also does NOT wait for a calendar date (that would be periodic retraining). It:

1. Receives **severity evidence** from both monitoring channels
2. Determines whether the evidence is strong enough to justify creating a candidate
3. Trains a candidate using **only the delayed-label buffer** (confirmed labels, recent data)
4. Passes the candidate through a **validation gate** against a chronological held-out window
5. Promotes if better, rejects if not — never blindly replaces production

---

### 1.5 ADWIN's Correct Role (Verified)

Research confirms: ADWIN is a **generic univariate stream detector**. When fed prediction errors (0 = correct, 1 = wrong), it requires confirmed labels. When fed feature values or model confidence scores, it does not.

In AdaptGuard AI's architecture:
- **ADWIN on error stream** → performance drift detector (requires delayed labels)
- **PSI/KS on feature stream** → covariate drift detector (label-free, immediate)

This dual application is correct and research-backed.

---

### 1.6 Oracle Experiment (New in v2)

A key addition to the experiment suite. The oracle provides the **upper bound reference** — it represents what perfect, zero-delay adaptation could achieve.

```
ORACLE:        Labels available at t=0 → maximum adaptation speed
ADAPTGUARD AI: Labels available at t=3 days → realistic
STATIC:        No adaptation → lower bound

"Cost of delay" = Oracle PR-AUC − AdaptGuard AI PR-AUC
```

This answers the question: *how much performance is sacrificed purely by the label-delay constraint?*

---

### 1.7 Periodic Retraining Baseline (Clarified in v2)

The spec now explicitly requires comparing retraining intervals: **1 day / 7 days / 14 days / 30 days**. Each interval is a separate experiment condition. The final selected interval for "Periodic Retraining" baseline should be chosen after preliminary experiments — it should be the one that performs best, to give the strongest possible comparison against AdaptiveML.

---

### 1.8 Statistical Evaluation (New in v2)

Results must not rely on a single run or single number. The v2 spec requires:
- Multiple experimental windows where applicable
- Bootstrap confidence intervals (95%)
- Mean ± standard deviation reported
- Statistical comparisons when justified

This elevates the work from an engineering report to a genuine research contribution.

---

### 1.9 Data Cleaning — Refined Rules (v2 Corrections)

Two important corrections from v1:

| Topic | v1 Assumption | v2 Correction |
|---|---|---|
| Timestamp gaps | "No gaps allowed" | Natural gaps are fine — validate format and ordering only |
| Duplicate TX IDs | Auto-delete | Detect → Investigate → Remove only confirmed duplicates |

These are important because blindly deleting or rejecting records could distort the temporal analysis.

---

### 1.10 PSI Threshold Caution (New in v2)

The spec explicitly states: **do not treat generic PSI thresholds (0.1 / 0.2) as universal truth.** The standard financial rule-of-thumb (PSI > 0.2 = significant shift) was developed for credit scoring, not streaming transaction fraud detection. Thresholds must be calibrated experimentally on this specific dataset. This is a notable research-honesty requirement.

---

### 1.11 Business Cost Metric (New in v2)

The spec adds a business cost metric with sensitivity analysis across FN:FP cost ratios:

```
Total Cost = FN × Cost_FN + FP × Cost_FP

Test ratios:
FN:FP = 10:1   (missing fraud is extremely costly)
FN:FP =  5:1   (moderately costly)
FN:FP =  2:1   (near-symmetric)
```

No real monetary values are invented. The ratios are the experimental variables.

---

### 1.12 Incremental Build Strategy (New in v2)

The spec now defines 4 explicit version targets:

| Version | Components | Purpose |
|---|---|---|
| **v1 — Core Research** | Dataset + Features + LR + XGBoost + ADWIN + SGD + Basic Controller | Prove the research loop works |
| **v2 — Research Depth** | + PSI + PH + KSWIN + MMD + Delayed Labels + Severity + Validation + Rollback + Ablation + Oracle | Full research contribution |
| **v3 — Application** | + FastAPI + Streamlit + MLflow + PostgreSQL | Demonstration layer |
| **v4 — Optional** | Cloud deployment | Production-readiness exploration |

This staged approach ensures the core science is working before adding engineering complexity.

---

## PART 2 — VERIFIED RESEARCH SOURCES

### 2.1 Primary Dataset & Simulator

| Resource | URL | Verification Status |
|---|---|---|
| ULB/Worldline Fraud Detection Handbook | https://fraud-detection-handbook.github.io/fraud-detection-handbook/ | ✅ Live — verified |
| Simulator Documentation | https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html | ✅ Live — verified |
| Handbook GitHub | https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook | ✅ Active repository |

**Simulator confirmed facts:**
- ~1,754,155 transactions, ~183 days, ~14,681 fraud cases
- Configurable compromised terminal scenarios → abrupt drift
- Configurable compromised customer scenarios → gradual drift
- All scenario timing is controlled → known drift-start timestamps for detection delay measurement

### 2.2 Academic Papers

| Paper | URL | Relevance |
|---|---|---|
| Le Borgne et al. — Streaming Active Learning for Credit Card Fraud Detection (2018) | https://arxiv.org/abs/1804.07481 | Streaming fraud, non-stationarity, delayed labels, class imbalance |
| Google Research — Learning Importance Under Concept Drift | https://research.google/blog/learning-the-importance-of-training-data-under-concept-drift/ | Drift-aware data weighting strategies |

### 2.3 Technology Libraries (Verified Active)

| Library | URL | Role |
|---|---|---|
| **River** | https://riverml.xyz/ | ADWIN, PH, KSWIN, online learners — all in one library |
| **Scikit-learn** | https://scikit-learn.org/ | SGDClassifier (partial_fit), baselines, prequential eval |
| **XGBoost** | https://xgboost.readthedocs.io/ | Strong static and periodic-retraining baseline |
| **MLflow** | https://mlflow.org/ | Experiment tracking, model registry, version lifecycle |
| **SHAP** | https://shap.readthedocs.io/ | Feature importance after drift events |
| **FastAPI** | https://fastapi.tiangolo.com/ | REST API (v3 phase) |
| **Streamlit** | https://streamlit.io/ | Real-time dashboard (v3 phase) |

### 2.4 Drift Detection — Research-Verified Summary

| Detector | Label Dependency | Primary Use | Key Finding |
|---|---|---|---|
| **ADWIN** | ✅ Needs labels (for error stream) | Performance drift detection | Self-tuning window; formal statistical guarantees; primary concept drift detector |
| **PSI** | ❌ Label-free | Feature-level distribution monitoring | Finance-standard; PSI thresholds must be calibrated per dataset — generic 0.1/0.2 thresholds are not universal |
| **KS Test** | ❌ Label-free | Univariate distribution comparison | Non-parametric; effective for continuous numerical features |
| **Page-Hinkley** | ✅ Needs labels (for error stream) | Abrupt mean shifts | Lightweight; requires careful λ/δ tuning |
| **KSWIN** | ❌ Label-free | Distribution-level windowed comparison | Can have high false positive rate if α/window not carefully tuned |
| **MMD** | ❌ Label-free | Multivariate silent drift | Advanced experiment; more computationally intensive |

### 2.5 Prequential Evaluation — Research-Verified

Research confirms this is the **standard protocol for streaming fraud detection**:

1. **Test first** → model predicts on transaction before label is known
2. **Label arrives** (after delay) → error recorded
3. **Train/update** → online/adaptive model updates with confirmed label
4. Repeat for every transaction in chronological order

**Why not holdout?** Standard holdout requires a static test set, which is incompatible with the streaming + delayed-label + concept drift setup. Prequential naturally handles all three.

### 2.6 Champion-Challenger Pattern — Research-Verified

The validation gate in AdaptGuard AI implements the **Champion-Challenger** pattern, which is research and industry standard for high-stakes model deployments:

- Challenger (candidate) runs through offline validation before any production exposure
- Promotion only when challenger beats champion by sufficient margin
- Clear audit trail for every model version (MLflow registry)
- Formal governance around the promotion criteria

---

## PART 3 — FULL IMPLEMENTATION PLAN

### 3.1 Directory Structure (Final)

```
AdaptGuard-AI/
│
├── data/
│   ├── raw/                    ← Simulator output CSV files (never modified)
│   ├── processed/              ← Cleaned, feature-engineered parquet files
│   └── README.md
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_drift_analysis.ipynb
│   └── 04_adaptation_experiments.ipynb
│
├── src/
│   ├── data/
│   │   ├── simulator.py        ← ULB/Worldline simulator interface
│   │   ├── loader.py           ← Data loading with temporal validation
│   │   └── validator.py        ← Cleaning rules (no blind dropna)
│   ├── features/
│   │   ├── time_features.py
│   │   ├── customer_features.py ← Rolling historical only (no global mean)
│   │   ├── terminal_features.py ← Historical fraud rate (label-time aware)
│   │   └── velocity_features.py ← 5min / 10min / 1h / 24h windows
│   ├── models/
│   │   ├── baseline.py         ← LogReg, RF, XGBoost wrappers
│   │   ├── online.py           ← SGDClassifier / River model wrappers
│   │   └── registry.py         ← Model versioning + rollback store
│   ├── drift/
│   │   ├── data_monitor.py     ← PSI, KS, MMD (label-free channel)
│   │   ├── perf_monitor.py     ← ADWIN, PH (label-required channel)
│   │   ├── severity.py         ← Multi-signal severity scoring
│   │   └── monitor.py          ← Orchestrates both channels
│   ├── adaptation/
│   │   ├── controller.py       ← Core adaptive controller
│   │   ├── candidate.py        ← Candidate model training + validation gate
│   │   └── delayed_labels.py   ← Label buffer + release simulation
│   ├── evaluation/
│   │   ├── metrics.py          ← PR-AUC, F1, F2, detection delay, etc.
│   │   ├── cost.py             ← Business cost with FN/FP ratios
│   │   ├── prequential.py      ← Test-then-train streaming protocol
│   │   ├── statistical.py      ← Bootstrap CIs, mean ± std
│   │   └── ablation.py         ← Ablation experiment runner
│   └── utils/
│       ├── logger.py
│       └── config.py
│
├── api/               ← Phase v3
│   └── main.py
├── dashboard/         ← Phase v3
│   └── app.py
├── experiments/
│   ├── e1_stable/
│   ├── e2_abrupt/
│   ├── e3_gradual/
│   ├── e4_recurring/
│   ├── e5_delayed_labels/
│   └── e6_safety/
├── models/            ← Serialized model files
├── results/           ← CSVs, plots, tables
├── configs/           ← YAML configs with pinned parameters
├── tests/
├── requirements.txt   ← Pinned exact versions (not "latest stable")
├── README.md
└── LICENSE
```

---

### 3.2 16-Phase Build Order

| Phase | Work | Version |
|---|---|---|
| 1 | Environment, simulator, data pipeline | v1 |
| 2 | Leak-free feature engineering | v1 |
| 3 | Baseline models (LR, RF, XGBoost) | v1 |
| 4 | Chronological / prequential evaluation framework | v1 |
| 5 | ADWIN (perf monitor) + PSI (data monitor) | v1 |
| 6 | Basic Adaptive Controller | v1 |
| 7 | Delayed label simulation (0/1/3/7 days) | v2 |
| 8 | Validation gate + rejection tracking | v2 |
| 9 | Rollback mechanism | v2 |
| 10 | Research experiments (E1–E6 + Oracle) | v2 |
| 11 | Ablation study (A1–A5) | v2 |
| 12 | Advanced detectors (PH, KSWIN, MMD) | v2 |
| 13 | FastAPI backend | v3 |
| 14 | Streamlit dashboard | v3 |
| 15 | MLflow complete integration | v3 |
| 16 | Final research report | v3 |

> [!NOTE]
> Phases 13–16 come **after** all research experiments are complete. The API and dashboard are the demonstration layer, not the research contribution.

---

### 3.3 Research Questions → Experiments → Metrics Map

| RQ | Question | Experiment | Metric |
|---|---|---|---|
| RQ1 | How quickly is drift detected? | E2, E3 | Detection Delay (tx) |
| RQ2 | Does dual-channel monitoring improve detection? | E2 ablation | False alarms, delay |
| RQ3 | Does severity improve decisions? | A2 ablation | PR-AUC, update count |
| RQ4 | Does selective adaptation reduce degradation? | E2, E3, E4 | PR-AUC over time |
| RQ5 | How do delayed labels affect adaptation? | E5 + Oracle | PR-AUC at each delay |
| RQ6 | Can validation gate prevent harmful promotions? | E6 | Rejection rate |
| RQ7 | Can rollback recover from harmful deployment? | E6 part 2 | Rollback rate, recovery |
| RQ8 | How does AdaptGuard compare to all baselines? | All experiments | Full metric table |

---

### 3.4 Experiment Results — All TBD

> [!CAUTION]
> No numbers are filled in before experiments are completed. No result is assumed in advance.

| Metric | Static XGBoost | Periodic Retrain | Always-Online | AdaptGuard AI |
|---|---|---|---|---|
| PR-AUC (E1) | TBD | TBD | TBD | TBD |
| PR-AUC (E2) | TBD | TBD | TBD | TBD |
| PR-AUC (E3) | TBD | TBD | TBD | TBD |
| Detection Delay | N/A | N/A | TBD | TBD |
| Adaptation Gain | N/A | TBD | TBD | TBD |
| Rejected Candidates | N/A | N/A | N/A | TBD |
| Rollback Count | N/A | N/A | N/A | TBD |
| Business Cost (10:1) | TBD | TBD | TBD | TBD |

---

*Report version 2.0 · Sources verified 2026-08-23 · All performance numbers are TBD until experiments complete*

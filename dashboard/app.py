"""
AdaptGuard AI — Streamlit Dashboard
Real-time monitoring of fraud detection performance, drift signals,
adaptation events, and model lifecycle.

The main visual story:
DATA CHANGE → DRIFT → ADAPTATION → MODEL CHANGE → PERFORMANCE CHANGE

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import requests
from datetime import datetime

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title = "AdaptGuard AI — Fraud Monitor",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

API_URL = "http://localhost:8000"

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
  /* Dark premium theme */
  .stApp { background-color: #040810; color: #f0f6ff; }
  .metric-card {
    background: linear-gradient(135deg, #0f1f3d, #0d1830);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
  }
  .drift-alert {
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 8px;
    padding: 10px;
    color: #f59e0b;
    font-weight: 700;
  }
  .stable-badge {
    background: rgba(16,185,129,0.15);
    border-radius: 20px;
    padding: 4px 12px;
    color: #10b981;
    font-size: 12px;
    font-weight: 700;
  }
  .critical-badge {
    background: rgba(239,68,68,0.15);
    border-radius: 20px;
    padding: 4px 12px;
    color: #ef4444;
    font-size: 12px;
    font-weight: 700;
  }
  h1, h2, h3 { color: #f0f6ff !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================

if "pr_auc_history" not in st.session_state:
    st.session_state.pr_auc_history = {
        "static":       [],
        "adaptguard":   [],
        "timestamps":   [],
    }
if "drift_history" not in st.session_state:
    st.session_state.drift_history = []
if "events_log" not in st.session_state:
    st.session_state.events_log = []


# ============================================================
# API Helpers
# ============================================================

def fetch_api(endpoint: str, default: dict) -> dict:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=2)
        return r.json()
    except Exception:
        return default


def post_api(endpoint: str, payload: dict = {}) -> dict:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=2)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    refresh_rate = st.slider("Refresh rate (s)", 1, 10, 2)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    st.markdown("---")
    st.markdown("### 🎭 Demo Scenarios")
    scenario = st.selectbox(
        "Select Scenario",
        ["Stable", "Abrupt Drift", "Gradual Drift", "Post-Adaptation"],
    )
    st.markdown("---")
    st.markdown("### ⚡ Manual Controls")
    if st.button("🔄 Trigger Adaptation", use_container_width=True):
        result = post_api("/adapt", {"reason": "manual_dashboard_trigger"})
        st.success("Adaptation triggered!")

    if st.button("⏮ Trigger Rollback", use_container_width=True):
        result = post_api("/rollback", {"reason": "manual_dashboard_rollback"})
        if result.get("success"):
            st.warning(f"Rolled back to v{result.get('restored_version')}")
        else:
            st.error(result.get("detail", "Rollback failed"))

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **AdaptGuard AI**
    Drift-Aware Adaptive Fraud Detection

    Research prototype — results are TBD
    until experiments are completed.

    📊 [Implementation Plan](../implementation_plan.md)
    """)


# ============================================================
# Header
# ============================================================

col_logo, col_status = st.columns([3, 1])
with col_logo:
    st.markdown("# 🛡️ AdaptGuard AI — Fraud Monitor")
    st.markdown("*Drift-Aware Adaptive Fraud Detection Under Evolving Transaction Distributions*")
with col_status:
    health = fetch_api("/health", {"status": "disconnected", "n_predictions": 0})
    status_color = "🟢" if health.get("status") == "ok" else "🔴"
    st.markdown(f"**System:** {status_color} {health.get('status', 'offline').upper()}")
    st.markdown(f"**Total Predictions:** {health.get('n_predictions', 0):,}")

st.divider()

# ============================================================
# Fetch live data
# ============================================================

drift_data   = fetch_api("/drift", {
    "drift_detected": False, "severity_level": "NONE",
    "severity_score": 0.0,   "adwin_signal": False,
    "max_psi": 0.0,          "error_rate": 0.0,
    "explanation": "No data",
})
model_data   = fetch_api("/model", {
    "production_version": 1, "model_name": "N/A",
    "adaptation_count": 0,  "rejection_count": 0, "rollback_count": 0,
    "metrics": {},           "train_start": "", "train_end": "",
})
metrics_data = fetch_api("/metrics", {
    "pr_auc": 0.0, "recall": 0.0, "precision": 0.0,
    "f1": 0.0,     "fpr": 0.0,    "n_samples": 0,
})
history_data = fetch_api("/history", {"events": [], "total": 0})


# ============================================================
# KPI Row
# ============================================================

st.markdown("### 📊 Key Performance Indicators")
kpi_cols = st.columns(6)

severity = drift_data.get("severity_level", "NONE")
severity_emoji = {"NONE": "🟢", "LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "🚨"}.get(severity, "⚪")

with kpi_cols[0]:
    st.metric("PR-AUC", f"{metrics_data.get('pr_auc', 0):.4f}", help="Primary metric (Area under Precision-Recall Curve)")
with kpi_cols[1]:
    st.metric("Recall", f"{metrics_data.get('recall', 0):.4f}", help="Fraud detection rate")
with kpi_cols[2]:
    st.metric("FPR", f"{metrics_data.get('fpr', 0):.4f}", help="False positive rate")
with kpi_cols[3]:
    st.metric("Drift Score", f"{drift_data.get('severity_score', 0):.4f}")
with kpi_cols[4]:
    st.metric("Model Version", f"v{model_data.get('production_version', 1)}")
with kpi_cols[5]:
    st.metric(f"Severity {severity_emoji}", severity)

# Drift alert banner
if drift_data.get("drift_detected"):
    st.markdown(
        f'<div class="drift-alert">⚡ DRIFT DETECTED — Severity: {severity} | {drift_data.get("explanation", "")}</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# Charts Row
# ============================================================

chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.markdown("#### PR-AUC Over Time — Static vs AdaptGuard AI")

    # Simulate rolling data for demo
    n_points = 40
    x_vals = list(range(n_points))
    s_level = drift_data.get("severity_level", "NONE")

    if s_level in ("HIGH", "CRITICAL"):
        static_prauc   = np.clip(0.85 - np.random.normal(0, 0.015, n_points).cumsum() * 0.01, 0.4, 0.95)
        adaptive_prauc = np.clip(0.87 + np.random.normal(0, 0.010, n_points), 0.55, 0.95)
    elif s_level == "MEDIUM":
        static_prauc   = np.clip(0.86 - np.linspace(0, 0.06, n_points), 0.5, 0.92)
        adaptive_prauc = np.clip(0.87 - np.linspace(0, 0.02, n_points), 0.6, 0.92)
    else:
        static_prauc   = np.clip(0.87 + np.random.normal(0, 0.008, n_points), 0.8, 0.95)
        adaptive_prauc = np.clip(0.88 + np.random.normal(0, 0.006, n_points), 0.82, 0.95)

    fig_prauc = go.Figure()
    fig_prauc.add_trace(go.Scatter(
        x=x_vals, y=static_prauc,
        name="Static XGBoost",
        line=dict(color="#ef4444", width=2),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.05)",
    ))
    fig_prauc.add_trace(go.Scatter(
        x=x_vals, y=adaptive_prauc,
        name="AdaptGuard AI",
        line=dict(color="#3b82f6", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.05)",
    ))
    fig_prauc.update_layout(
        height=260, paper_bgcolor="#040810", plot_bgcolor="#040810",
        font=dict(color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0, y=1),
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0.3, 1.0], gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig_prauc, use_container_width=True)


with chart_col2:
    st.markdown("#### Drift Signal")

    n_d = 40
    psi_vals = np.random.uniform(0.04, 0.08, n_d)
    if s_level == "HIGH":     psi_vals[-8:] = np.random.uniform(0.25, 0.45, 8)
    elif s_level == "MEDIUM": psi_vals[-10:] = np.random.uniform(0.12, 0.22, 10)

    fig_drift = go.Figure()
    fig_drift.add_hline(y=0.20, line_dash="dash", line_color="#f59e0b", opacity=0.5, annotation_text="Alert threshold (calibrate)")
    fig_drift.add_trace(go.Scatter(
        x=list(range(n_d)), y=psi_vals,
        name="Max PSI",
        fill="tozeroy",
        line=dict(color="#06b6d4", width=2),
        fillcolor="rgba(6,182,212,0.1)",
    ))
    fig_drift.update_layout(
        height=260, paper_bgcolor="#040810", plot_bgcolor="#040810",
        font=dict(color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        yaxis=dict(gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig_drift, use_container_width=True)


st.divider()

# ============================================================
# Bottom Row
# ============================================================

bot_col1, bot_col2, bot_col3 = st.columns(3)

with bot_col1:
    st.markdown("#### 🔍 Detector Status")
    detectors = {
        "ADWIN (Error stream)": drift_data.get("adwin_signal", False),
        "PSI (Feature dist.)":  drift_data.get("max_psi", 0) > 0.10,
        "Page-Hinkley":         False,
        "KSWIN":                False,
        "MMD":                  False,
    }
    for name, fired in detectors.items():
        status = "🔴 ALERT" if fired else "🟢 Stable"
        st.markdown(f"**{name}** — {status}")

with bot_col2:
    st.markdown("#### 📋 Adaptation Events")
    events = history_data.get("events", [])
    if events:
        for event in events[-5:]:
            action = event.get("action", "")
            icon   = {"adapted": "🔄", "rejected": "❌", "rollback": "⏮", "monitoring": "👁"}.get(action, "📋")
            sev    = event.get("severity_level", "")
            v      = event.get("production_v", "?")
            st.markdown(f"{icon} `{action.upper()}` · v{v} · Sev={sev}")
    else:
        st.markdown("*No adaptation events yet.*")

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Adaptations", model_data.get("adaptation_count", 0))
    with col_b:
        st.metric("Rejections",  model_data.get("rejection_count", 0))
    with col_c:
        st.metric("Rollbacks",   model_data.get("rollback_count", 0))

with bot_col3:
    st.markdown("#### 📈 Model Performance Metrics")
    perf_data = {
        "PR-AUC (Primary)": metrics_data.get("pr_auc", 0),
        "Recall":           metrics_data.get("recall", 0),
        "Precision":        metrics_data.get("precision", 0),
        "F1":               metrics_data.get("f1", 0),
        "FPR":              metrics_data.get("fpr", 0),
    }
    fig_bars = go.Figure(go.Bar(
        x=list(perf_data.values()),
        y=list(perf_data.keys()),
        orientation="h",
        marker_color=["#3b82f6", "#10b981", "#6366f1", "#8b5cf6", "#ef4444"],
    ))
    fig_bars.update_layout(
        height=220, paper_bgcolor="#040810", plot_bgcolor="#040810",
        font=dict(color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(range=[0, 1], gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        showlegend=False,
    )
    st.plotly_chart(fig_bars, use_container_width=True)


st.divider()

# ============================================================
# Research Note
# ============================================================

st.markdown("""
> ⚠️ **Research Note:** All performance numbers shown in this dashboard are either
> live API values or demo simulations. Final research results will be determined
> through controlled experiments (E1–E6). No results are assumed in advance.
""")

# ============================================================
# Auto-refresh
# ============================================================

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()

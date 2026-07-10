"""Forecast — Per-region 7-day forecasts with confidence bands + production model metrics."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: F401  bootstrap sys.path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import (
    empty_state,
    get_forecast_for_region,
    get_production_model_metrics,
    get_silver_events_per_country_day,
    page_header,
    sidebar_status_panel,
)

st.set_page_config(page_title="OSCAR · Forecast", page_icon="📈", layout="wide")
sidebar_status_panel()

page_header(
    "Forecast",
    "7-day event-volume forecasts per region with 95% confidence bands.",
)


def _has_silver() -> bool:
    return not get_silver_events_per_country_day().empty


def _list_regions() -> list[str]:
    forecasts_dir = Path("data/processed/forecasts")
    if not forecasts_dir.exists():
        return []
    return sorted(p.stem.replace("forecast_", "") for p in forecasts_dir.glob("forecast_*.parquet"))


if not _has_silver():
    empty_state(
        "No silver table yet.",
        "Run `python -m src.ingestion.cli transform` first.",
    )
    st.stop()

regions = _list_regions()
if not regions:
    empty_state(
        "No forecasts persisted yet.",
        "Run `python -m src.ml.cli forecast --regions UKR --periods 7` to generate them.",
    )
    st.stop()

region = st.sidebar.selectbox("Region", regions, key="forecast_region")
fc = get_forecast_for_region(region)
if fc.empty:
    empty_state(f"No forecast for {region}.")
    st.stop()

st.markdown(f"### {region} · 7-day event forecast")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=fc["ds"],
        y=fc["yhat"],
        mode="lines+markers",
        line={"color": "#1e293b", "width": 2},
        marker={"size": 8, "color": "#3b82f6"},
        name="Forecast (median)",
    )
)
fig.add_trace(
    go.Scatter(
        x=pd.concat([fc["ds"], fc["ds"][::-1]]),
        y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.12)",
        line={"color": "rgba(0,0,0,0)"},
        hoverinfo="skip",
        name="95% CI",
    )
)
fig.update_layout(
    height=420,
    yaxis_title="Predicted event count",
    xaxis_title="",
    showlegend=True,
    margin={"l": 0, "r": 0, "t": 10, "b": 0},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend={"orientation": "h", "y": -0.15, "yanchor": "top"},
    hovermode="x unified",
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("")
st.markdown("### Forecast values")
st.dataframe(
    fc.rename(
        columns={
            "ds": "date",
            "yhat": "predicted",
            "yhat_lower": "ci_low",
            "yhat_upper": "ci_high",
        }
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        "predicted": st.column_config.NumberColumn("Median", format="%.2f"),
        "ci_low": st.column_config.NumberColumn("CI low", format="%.2f"),
        "ci_high": st.column_config.NumberColumn("CI high", format="%.2f"),
    },
)

# ── Production model metrics ────────────────────────────
st.markdown("---")
st.markdown("### Production escalation models")
metrics = get_production_model_metrics()
configured = [
    ("escalation_h1", "1-day escalation classifier"),
    ("escalation_h3", "3-day escalation classifier"),
    ("escalation_h7", "7-day escalation classifier"),
]
for name, label in configured:
    m = metrics.get(name)
    if m is None:
        with st.expander(label, expanded=False):
            st.caption(
                "No Production version yet. Run `python -m src.ml.cli train-escalation --horizon 7 --promote`."
            )
        continue
    with st.expander(label, expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Stage", m["stage"])
        acc = m["metrics"].get("accuracy", 0)
        f1 = m["metrics"].get("f1", 0)
        pr = m["metrics"].get("pr_auc", 0)
        c2.metric("PR-AUC", f"{pr:.3f}")
        c3.metric("F1 / Accuracy", f"{f1:.3f} / {acc:.3f}")

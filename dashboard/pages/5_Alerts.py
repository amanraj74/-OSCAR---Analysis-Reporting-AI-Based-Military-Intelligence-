"""Alerts — Live anomaly feed with severity sort and detail view."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # noqa: F401  bootstrap sys.path
import pandas as pd
import streamlit as st

from dashboard.utils import (
    empty_state,
    fmt_dt,
    get_recent_anomalies,
    page_header,
    severity_color,
    sidebar_status_panel,
)

st.set_page_config(page_title="OSCAR · Alerts", page_icon="⚠️", layout="wide")
sidebar_status_panel()

page_header(
    "Alerts",
    "Live anomaly feed. Severity normalized 0–1; higher = more anomalous.",
)


def render() -> None:
    with st.sidebar:
        st.markdown("### Filters")
        limit = st.slider("Show last N", 10, 200, 50, key="alerts_limit")
        min_severity = st.slider("Min severity", 0.0, 1.0, 0.0, 0.05, key="alerts_min_sev")

    df = get_recent_anomalies(limit=limit)

    if df.empty or "severity" not in df.columns:
        empty_state(
            "No anomalies yet.",
            "Run `python -m src.ml.cli detect-anomalies` to generate.",
        )
        return

    df = df[df["severity"] >= min_severity].copy()
    if "detected_at" in df.columns:
        df["detected_at"] = pd.to_datetime(df["detected_at"])

    st.markdown(f"### {len(df)} alert(s) above severity {min_severity:.2f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Critical (>= 0.7)", int((df["severity"] >= 0.7).sum()))
    col2.metric(
        "High (0.4 - 0.7)",
        int(((df["severity"] >= 0.4) & (df["severity"] < 0.7)).sum()),
    )
    col3.metric("Medium (< 0.4)", int((df["severity"] < 0.4).sum()))

    st.divider()

    sort_by = st.selectbox(
        "Sort by",
        ["severity_desc", "severity_asc", "newest", "oldest", "region"],
        format_func=lambda v: {
            "severity_desc": "Severity (high -> low)",
            "severity_asc": "Severity (low -> high)",
            "newest": "Newest first",
            "oldest": "Oldest first",
            "region": "Region (A->Z)",
        }[v],
        key="alerts_sort",
    )

    if sort_by == "severity_desc":
        df = df.sort_values("severity", ascending=False)
    elif sort_by == "severity_asc":
        df = df.sort_values("severity", ascending=True)
    elif sort_by == "newest":
        df = df.sort_values("detected_at", ascending=False)
    elif sort_by == "oldest":
        df = df.sort_values("detected_at", ascending=True)
    elif sort_by == "region":
        df = df.sort_values("region")

    display = df[
        ["region", "date", "anomaly_type", "severity", "score", "description", "detected_at"]
    ].copy()
    display["severity"] = display["severity"].map(lambda v: f"{v:.3f}")
    display["score"] = display["score"].map(lambda v: f"{v:.3f}")

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "detected_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )

    st.divider()
    st.markdown("### Anomaly detail")

    selected = st.selectbox(
        "Inspect",
        df.index,
        format_func=lambda i: (
            f"{df.loc[i, 'region']} . {df.loc[i, 'anomaly_type']} . "
            f"sev={df.loc[i, 'severity']:.2f} . {df.loc[i, 'date']}"
        ),
        key="alerts_detail",
    )
    row = df.loc[selected]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"**Region** {row['region']}  \n"
            f"**Date** {row['date']}  \n"
            f"**Type** {row['anomaly_type']}  \n"
            f"**Severity** "
            f"<span style='color:{severity_color(row['severity'])};font-weight:600;'>"
            f"{row['severity']:.3f}</span>  \n"
            f"**Score** {row['score']:.3f}  \n"
            f"**Detected** {fmt_dt(row['detected_at'])}",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("**Description**")
        st.write(row["description"] or "-")
        ctx = row.get("context")
        if isinstance(ctx, dict) and ctx:
            st.json(ctx)
        elif isinstance(ctx, str) and ctx:
            try:
                st.json(json.loads(ctx))
            except Exception:
                st.text(ctx)


render()

"""
dashboard/tabs/performance.py
Rolling FPS, latency, and decision throughput.
"""

import streamlit as st
from dashboard.components import performance_chart


def render(snapshot: dict) -> None:
    """Render the performance tab."""
    st.subheader("System Performance")
    st.caption(
        "Rolling FPS, agent pipeline latency, and per-decision-type counts "
        "updated on every dashboard refresh."
    )
    performance_chart.render(snapshot)
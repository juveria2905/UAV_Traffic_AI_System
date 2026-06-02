"""
dashboard/tabs/reasoning.py
Reasoning chain browser.
"""

import streamlit as st
from dashboard.components import reasoning_timeline


def render(snapshot: dict) -> None:
    """Render the reasoning chains tab."""
    st.subheader("Reasoning Chains")
    st.caption(
        "Reasoning chains record the step-by-step logic behind every "
        "REROUTE and EMERGENCY_STOP decision, enabling full explainability."
    )
    reasoning_timeline.render(snapshot)
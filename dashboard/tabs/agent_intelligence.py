"""
dashboard/tabs/agent_intelligence.py
Agent hierarchy, communication graph, and confidence gauge.
"""

import streamlit as st
from dashboard.components import agent_hierarchy, communication_graph, confidence_gauge


def render(snapshot: dict, api_base: str) -> None:
    """Render the agent intelligence tab."""
    st.subheader("Agent System Intelligence")
    st.caption(
        "Hierarchical agent decisions, inter-agent communication, "
        "and confidence analysis."
    )

    # Three-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hierarchy")
        agent_hierarchy.render(snapshot)

    with col2:
        st.subheader("Communication")
        communication_graph.render(snapshot)

    st.divider()

    st.subheader("Decision Confidence")
    confidence_gauge.render(snapshot)

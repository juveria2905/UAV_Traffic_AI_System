"""
dashboard/components/confidence_gauge.py
Displays per-track confidence breakdowns from agent decisions.
Reads from snapshot["agent_decisions"] list of AgentDecision.to_dict():
  agent, track_id, action, reason, confidence, frame_idx, priority,
  timestamp, chain_id
"""

import streamlit as st
import pandas as pd
from typing import List


def render(snapshot: dict) -> None:
    """Render confidence distribution for active agent decisions."""
    decisions: List[dict] = snapshot.get("agent_decisions", [])

    if not decisions:
        st.info("No agent decisions available.")
        return

    # Non-MONITOR decisions are most interesting
    non_monitor = [d for d in decisions if d.get("action", "MONITOR") != "MONITOR"]
    all_dec      = non_monitor if non_monitor else decisions

    # Build summary
    rows = []
    for d in all_dec:
        rows.append({
            "Track":    d.get("track_id", "?"),
            "Action":   d.get("action", "MONITOR"),
            "Conf":     round(d.get("confidence", 0), 3),
            "Priority": d.get("priority", 1),
            "Agent":    d.get("agent", "?"),
            "Reason":   (d.get("reason", "") or "")[:60],
        })

    df = pd.DataFrame(rows).sort_values("Conf", ascending=False)

    # Confidence distribution chart
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        action_colors = {
            "EMERGENCY_STOP": "#ff3333",
            "REROUTE":        "#ff8800",
            "HOLD":           "#ffcc00",
            "MONITOR":        "#44cc44",
            "PRIORITIZE":     "#cc44ff",
        }

        for action, grp in df.groupby("Action"):
            fig.add_trace(go.Bar(
                name=action,
                x=[f"#{r['Track']}" for _, r in grp.iterrows()],
                y=grp["Conf"].tolist(),
                marker_color=action_colors.get(action, "#888888"),
                text=[f"{c:.2f}" for c in grp["Conf"].tolist()],
                textposition="auto",
            ))

        fig.update_layout(
            title="Decision Confidence by Track",
            xaxis_title="Track ID",
            yaxis_title="Confidence",
            yaxis_range=[0, 1],
            barmode="group",
            height=280,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.dataframe(df[["Track", "Action", "Conf"]], use_container_width=True, hide_index=True)

    # Detailed table
    with st.expander("Confidence details", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # System-level confidence scorer weights if available
    system_state = snapshot.get("system_state", {})
    goal_summary = system_state.get("goal_summary", {})
    if goal_summary:
        biases = goal_summary.get("action_biases", {})
        if biases:
            st.caption("Goal-driven action biases")
            bias_df = pd.DataFrame([
                {"Action": k, "Bias": round(v, 4)} for k, v in biases.items()
            ])
            st.dataframe(bias_df, use_container_width=True, hide_index=True)
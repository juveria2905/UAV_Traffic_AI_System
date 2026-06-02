"""
dashboard/components/memory_timeline.py
Shows DecisionMemory stats from system_state["memory_stats"].
memory_stats matches DecisionMemory.get_global_stats():
  total_records_stored, active_track_memories, total_decisions,
  global_action_counts, repeat_offenders, erratic_tracks,
  avg_success_rate, uptime_s
"""

import streamlit as st
import pandas as pd
from typing import Dict


ACTION_COLORS = {
    "EMERGENCY_STOP": "#ff3333",
    "REROUTE":        "#ff8800",
    "HOLD":           "#ffcc00",
    "MONITOR":        "#44cc44",
    "PRIORITIZE":     "#cc44ff",
}


def render(snapshot: dict) -> None:
    """Render memory system statistics and action distribution."""
    system_state  = snapshot.get("system_state", {})
    memory_stats: dict = system_state.get("memory_stats", {})

    if not memory_stats:
        st.info("Memory system not active or no data yet.")
        return

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records stored",    memory_stats.get("total_records_stored", 0))
    c2.metric("Active memories",   memory_stats.get("active_track_memories", 0))
    c3.metric("Total decisions",   memory_stats.get("total_decisions", 0))
    c4.metric("Repeat offenders",  memory_stats.get("repeat_offenders", 0))
    c5.metric("Avg success rate",  f"{memory_stats.get('avg_success_rate', 0):.1%}")

    st.divider()

    # Action distribution chart
    action_counts: Dict[str, int] = memory_stats.get("global_action_counts", {})
    if action_counts:
        st.subheader("Lifetime Action Distribution")
        try:
            import plotly.graph_objects as go

            labels = list(action_counts.keys())
            values = list(action_counts.values())
            colors = [ACTION_COLORS.get(l, "#888888") for l in labels]

            fig = go.Figure(go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                hole=0.4,
                textinfo="label+percent",
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            df = pd.DataFrame([
                {"Action": k, "Count": v}
                for k, v in action_counts.items()
            ])
            st.bar_chart(df.set_index("Action"))

    # Flags
    erratic = memory_stats.get("erratic_tracks", 0)
    if erratic:
        st.warning(f"⚡ {erratic} erratic track(s) detected in memory.")

    repeat = memory_stats.get("repeat_offenders", 0)
    if repeat:
        st.error(f"🔁 {repeat} repeat offender(s) tracked.")

    uptime = memory_stats.get("uptime_s", 0)
    st.caption(f"Memory system uptime: {uptime:.0f}s")
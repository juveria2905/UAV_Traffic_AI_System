"""
dashboard/components/communication_graph.py
Visualises inter-agent message bus activity.
Data from system_state (MessageBus.get_stats()):
  total_published, total_delivered, queue_pending,
  subscriber_count, topic_counts, history_size
"""

import streamlit as st
import pandas as pd
from typing import Dict


def render(snapshot: dict) -> None:
    """Render message bus activity and topic distribution."""
    system_state = snapshot.get("system_state", {})

    if not system_state:
        st.info("System state not available.")
        _render_agent_topology()
        return

    # Message bus stats are embedded in system_state if the pipeline exposes them
    # They come from HierarchicalAgentSystem.get_message_bus_stats()
    # which returns AgentMessageBus.get_stats()
    bus_stats: dict = system_state.get("message_bus_stats", {})

    if not bus_stats:
        st.info("Message bus statistics not available. "
                "The pipeline may not be exposing bus stats in the state update.")
        _render_agent_topology()
        return

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Published",   bus_stats.get("total_published", 0))
    c2.metric("Delivered",   bus_stats.get("total_delivered", 0))
    c3.metric("Subscribers", bus_stats.get("subscriber_count", 0))
    c4.metric("Queue depth", bus_stats.get("queue_pending", 0))

    # Topic distribution
    topic_counts: Dict[str, int] = bus_stats.get("topic_counts", {})
    if topic_counts:
        st.subheader("Message Topics")
        rows = [{"Topic": t, "Count": n} for t, n in sorted(
            topic_counts.items(), key=lambda x: x[1], reverse=True
        )]
        df = pd.DataFrame(rows)

        try:
            import plotly.express as px
            fig = px.bar(
                df, x="Count", y="Topic", orientation="h",
                title="Messages per topic",
                color="Count",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                height=max(200, len(rows) * 28),
                margin=dict(l=10, r=10, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.03)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(df, use_container_width=True, hide_index=True)

    _render_agent_topology()


def _render_agent_topology() -> None:
    """Static agent communication topology diagram."""
    st.subheader("Agent Communication Topology")
    st.markdown("""
    ```
    MonitorAgent-Zone-Alpha
        │  (local decisions)
        ▼
    CoordinatorAgent ◄── ConflictResolver
        │  (coordinated decisions)
        ▼
    ExecutorAgent
        │  (commands + log)
        ▼
    MessageBus ◄──────────────────────────────────┐
        │                                          │
        ├─► ZoneAgent-Z00..Z15  (via ZONE_*)       │
        │       │                                   │
        │       ▼                                   │
        │   DroneAgent-{track_id}                   │
        │                                           │
        └─► CityAgent (reads congested_zones) ──────┘
    ```
    """)

    st.caption(
        "Agents publish to Topics (collision.*, decision.*, zone.*). "
        "Subscribers receive messages at the next flush() call (once per frame)."
    )
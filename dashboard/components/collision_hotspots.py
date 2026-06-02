"""
dashboard/components/collision_hotspots.py
Scatter chart showing collision hotspot positions in frame space.
Uses collision_x/collision_y from CollisionEvent.to_dict().
"""

import streamlit as st
import pandas as pd
from typing import List


RISK_COLOR_MAP = {
    "HIGH":   "#ff3333",
    "MEDIUM": "#ff9900",
    "LOW":    "#ffdd00",
}


def render(snapshot: dict, frame_w: int = 1280, frame_h: int = 720) -> None:
    """Render a scatter plot of collision hotspot positions."""
    events: List[dict] = snapshot.get("collision_events", [])

    system_state = snapshot.get("system_state", {})
    city_state   = system_state.get("city_state", {})
    grid_info    = city_state.get("grid_info", {})
    actual_w     = grid_info.get("zone_w", frame_w / 4) * grid_info.get("cols", 4)
    actual_h     = grid_info.get("zone_h", frame_h / 4) * grid_info.get("rows", 4)

    # Filter events with known collision point
    hotspots = [
        ev for ev in events
        if ev.get("collision_x") is not None and ev.get("collision_y") is not None
    ]

    if not hotspots:
        st.info("No collision hotspot data available (no events with predicted collision points).")
        return

    rows = []
    for ev in hotspots:
        rows.append({
            "x":    ev["collision_x"],
            "y":    ev["collision_y"],
            "risk": ev.get("risk_level", "LOW"),
            "ttc":  round(ev.get("ttc_seconds", 0), 2),
            "pair": f"#{ev.get('track_id_a')} ↔ #{ev.get('track_id_b')}",
        })

    df = pd.DataFrame(rows)

    try:
        import plotly.express as px

        color_map = {k: v for k, v in RISK_COLOR_MAP.items()}
        fig = px.scatter(
            df,
            x="x", y="y",
            color="risk",
            color_discrete_map=color_map,
            size_max=20,
            hover_data=["ttc", "pair"],
            title="Collision Hotspots",
            labels={"x": "Frame X (px)", "y": "Frame Y (px)", "risk": "Risk Level"},
        )
        fig.update_layout(
            xaxis_range=[0, actual_w or frame_w],
            yaxis_range=[actual_h or frame_h, 0],  # flip Y to match image coords
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback to simple table if plotly not available
        st.dataframe(df, use_container_width=True, hide_index=True)
"""
dashboard/tabs/analytics.py
Collision hotspots, trajectory heatmap, and detection statistics.
"""

import streamlit as st
import pandas as pd
from typing import List

from dashboard.components import collision_hotspots, trajectory_heatmap


def render(snapshot: dict) -> None:
    """Render the analytics tab."""

    # ── Detection stats ───────────────────────────────────────────────────────
    st.subheader("Detection Statistics")
    detections: List[dict] = snapshot.get("detections", [])
    tracks:     List[dict] = snapshot.get("tracks", [])

    if not detections and not tracks:
        st.info("No detection or tracking data yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Detections (frame)", len(detections))
    c2.metric("Active tracks",      len(tracks))

    # Class distribution
    class_counts: dict = {}
    for t in tracks:
        cls = t.get("class_name", t.get("class", "unknown"))
        class_counts[cls] = class_counts.get(cls, 0) + 1
    if class_counts:
        c3.metric("Classes", len(class_counts))
        rows = [{"Class": k, "Count": v} for k, v in class_counts.items()]
        df   = pd.DataFrame(rows)
        if not df.empty:
            try:
                import plotly.express as px
                fig = px.pie(df, names="Class", values="Count",
                             title="Track class distribution")
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(df.set_index("Class"))
    else:
        c3.metric("Classes", 0)

    st.divider()

    # ── Trajectory heatmap ────────────────────────────────────────────────────
    st.subheader("Track Density Heatmap")
    trajectory_heatmap.render(snapshot)

    st.divider()

    # ── Collision hotspots ────────────────────────────────────────────────────
    st.subheader("Collision Hotspots")
    collision_hotspots.render(snapshot)

    st.divider()

    # ── Speed distribution ────────────────────────────────────────────────────
    st.subheader("Speed Distribution")
    speeds = []
    for t in tracks:
        vx = t.get("vx", 0)
        vy = t.get("vy", 0)
        s  = (vx**2 + vy**2) ** 0.5
        speeds.append({"Track": t.get("track_id", "?"), "Speed (px/f)": round(s, 2)})

    if speeds:
        spd_df = pd.DataFrame(speeds)
        try:
            import plotly.express as px
            fig = px.histogram(
                spd_df, x="Speed (px/f)", nbins=20,
                title="Speed distribution across active tracks",
                color_discrete_sequence=["#4488ff"],
            )
            fig.update_layout(
                height=220,
                margin=dict(l=40, r=20, t=40, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.03)",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(spd_df.set_index("Track"))
    else:
        st.info("No speed data available.")
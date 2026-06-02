"""
dashboard/components/trajectory_heatmap.py
Density heatmap of track positions accumulated across frames.
Uses cx/cy from snapshot["tracks"] to build a rolling position map.
"""

import streamlit as st
import numpy as np
from collections import deque
from typing import List


_MAX_POINTS = 2000


def _init_points() -> deque:
    if "heatmap_pts" not in st.session_state:
        st.session_state["heatmap_pts"] = deque(maxlen=_MAX_POINTS)
    return st.session_state["heatmap_pts"]


def render(snapshot: dict, frame_w: int = 1280, frame_h: int = 720) -> None:
    """Accumulate track positions and render a 2-D heatmap."""
    tracks: List[dict] = snapshot.get("tracks", [])
    pts: deque         = _init_points()

    for t in tracks:
        cx = t.get("cx", 0)
        cy = t.get("cy", 0)
        if cx and cy:
            pts.append((cx, cy))

    if not pts:
        st.info("Accumulating track positions for heatmap…")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    try:
        import plotly.graph_objects as go

        fig = go.Figure(go.Histogram2dContour(
            x=xs,
            y=ys,
            colorscale="Hot",
            reversescale=True,
            showscale=True,
            contours=dict(showlines=False),
            ncontours=20,
        ))
        # Overlay scatter for current frame
        cur_x = [t.get("cx", 0) for t in tracks]
        cur_y = [t.get("cy", 0) for t in tracks]
        if cur_x:
            fig.add_trace(go.Scatter(
                x=cur_x, y=cur_y,
                mode="markers",
                marker=dict(color="cyan", size=6, opacity=0.7),
                name="Current frame",
            ))

        fig.update_layout(
            title="Track density heatmap",
            xaxis=dict(title="X (px)", range=[0, frame_w]),
            yaxis=dict(title="Y (px)", range=[frame_h, 0]),  # flip Y
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback: numpy histogram rendered as image
        W, H   = 128, 72
        hist, _, _ = np.histogram2d(
            xs, ys,
            bins=[W, H],
            range=[[0, frame_w], [0, frame_h]],
        )
        hist_norm = (hist / hist.max() * 255).astype(np.uint8).T
        st.image(hist_norm, caption="Track density (fallback)", use_column_width=True)

    st.caption(f"Accumulated {len(pts)} position samples across last frames.")

    if st.button("Clear heatmap", key="clear_heatmap"):
        st.session_state["heatmap_pts"] = deque(maxlen=_MAX_POINTS)
        st.rerun()
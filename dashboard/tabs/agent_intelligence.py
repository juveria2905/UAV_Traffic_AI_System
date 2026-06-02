"""
dashboard/tabs/live_feed.py
Live annotated video frame + current collision alerts + active tracks.
"""

import streamlit as st
import base64
from typing import Optional

from dashboard.components import collision_alerts, track_viewer


def render(snapshot: dict, api_base: str) -> None:
    """Render the live feed tab."""
    col_frame, col_info = st.columns([3, 2])

    with col_frame:
        st.subheader("Live Frame")
        # Prefer the base64 frame from the /api/frame endpoint
        frame_b64: Optional[str] = None

        try:
            import requests
            resp = requests.get(f"{api_base}/api/frame", timeout=0.5)
            if resp.ok:
                data      = resp.json()
                frame_b64 = data.get("image_b64")
        except Exception:
            pass

        if frame_b64:
            st.markdown(
                f'<img src="{frame_b64}" style="width:100%;border-radius:6px"/>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Waiting for first frame from pipeline…")

        # System status bar
        fps       = snapshot.get("fps", 0)
        frame_idx = snapshot.get("frame_idx", 0)
        is_run    = snapshot.get("is_running", False)
        status_col = "🟢" if is_run else "🔴"
        st.caption(f"{status_col} Frame: {frame_idx} | FPS: {fps:.1f}")

    with col_info:
        st.subheader("Collision Alerts")
        collision_alerts.render(snapshot)

        st.divider()
        st.subheader("Active Tracks")
        track_viewer.render(snapshot)
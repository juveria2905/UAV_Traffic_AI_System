"""
dashboard/components/collision_alerts.py
Displays active collision risk events from API snapshot.
collision_events list items match CollisionEvent.to_dict() structure:
  frame_idx, track_id_a, track_id_b, ttc_frames, ttc_seconds,
  min_distance, risk_level, collision_x, collision_y
"""

import streamlit as st
import pandas as pd
from typing import List, Dict


RISK_COLORS = {
    "HIGH":   "#ff3333",
    "MEDIUM": "#ff9900",
    "LOW":    "#ffdd00",
    "NONE":   "#888888",
}

RISK_ICONS = {
    "HIGH":   "🔴",
    "MEDIUM": "🟠",
    "LOW":    "🟡",
    "NONE":   "⚪",
}


def render(snapshot: dict) -> None:
    """Render collision risk events table and alert banners."""
    events: List[dict] = snapshot.get("collision_events", [])

    high_risk   = [e for e in events if e.get("risk_level") == "HIGH"]
    medium_risk = [e for e in events if e.get("risk_level") == "MEDIUM"]
    low_risk    = [e for e in events if e.get("risk_level") == "LOW"]

    # Alert banners for HIGH risk
    if high_risk:
        st.error(f"⚠️ {len(high_risk)} HIGH-RISK collision(s) detected!")
        for ev in high_risk[:3]:
            st.error(
                f"Track #{ev['track_id_a']} ↔ #{ev['track_id_b']} — "
                f"TTC: {ev['ttc_seconds']:.2f}s — "
                f"Distance: {ev['min_distance']:.0f}px"
            )
    elif medium_risk:
        st.warning(f"⚠️ {len(medium_risk)} MEDIUM-RISK collision(s) detected.")
    else:
        st.success("✅ No high-risk collisions detected.")

    if not events:
        st.info("No collision events this frame.")
        return

    # Build table
    rows = []
    for ev in events:
        rows.append({
            "Risk":      RISK_ICONS.get(ev.get("risk_level", ""), "") + " " + ev.get("risk_level", "?"),
            "Track A":   ev.get("track_id_a", "?"),
            "Track B":   ev.get("track_id_b", "?"),
            "TTC (s)":   round(ev.get("ttc_seconds", 0), 2),
            "TTC (f)":   round(ev.get("ttc_frames", 0), 1),
            "Dist (px)": round(ev.get("min_distance", 0), 0),
            "Frame":     ev.get("frame_idx", "?"),
        })

    df = pd.DataFrame(rows)

    def _style_risk(val: str):
        for risk, color in RISK_COLORS.items():
            if risk in val:
                return f"color: {color}; font-weight: bold;"
        return ""

    styled = df.style.applymap(_style_risk, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", len(events))
    c2.metric("🔴 HIGH",   len(high_risk))
    c3.metric("🟠 MEDIUM", len(medium_risk))
    c4.metric("🟡 LOW",    len(low_risk))
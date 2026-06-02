"""
dashboard/components/track_viewer.py
Active track table with decision overlays.
Reads from API snapshot: tracks list and agent_decisions list.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any


DECISION_COLORS = {
    "EMERGENCY_STOP": "#ff4444",
    "REROUTE":        "#ff8800",
    "HOLD":           "#ffcc00",
    "MONITOR":        "#44cc44",
    "PRIORITIZE":     "#cc44ff",
}


def render(snapshot: dict) -> None:
    """Render active tracks with their current agent decisions."""
    tracks:    List[dict] = snapshot.get("tracks", [])
    decisions: List[dict] = snapshot.get("agent_decisions", [])

    # Build decision lookup by track_id
    dec_by_track: Dict[int, dict] = {}
    for d in decisions:
        tid = d.get("track_id")
        if tid is not None:
            dec_by_track[tid] = d

    if not tracks:
        st.info("No active tracks.")
        return

    rows = []
    for t in tracks:
        tid    = t.get("track_id", t.get("id", "?"))
        cls    = t.get("class_name", t.get("class", "unknown"))
        cx     = round(t.get("cx", 0), 1)
        cy     = round(t.get("cy", 0), 1)
        vx     = round(t.get("vx", 0), 2)
        vy     = round(t.get("vy", 0), 2)
        speed  = round((vx**2 + vy**2) ** 0.5, 2)
        conf   = round(t.get("confidence", 0), 2)
        age    = t.get("age_frames", 0)

        dec    = dec_by_track.get(tid, {})
        action = dec.get("action", "MONITOR")
        a_conf = round(dec.get("confidence", 0.0), 2)
        reason = dec.get("reason", "")

        rows.append({
            "ID":       tid,
            "Class":    cls,
            "cx":       cx,
            "cy":       cy,
            "Speed":    speed,
            "Conf":     conf,
            "Age":      age,
            "Action":   action,
            "A.Conf":   a_conf,
            "Reason":   reason[:60] if reason else "",
        })

    df = pd.DataFrame(rows)

    # Colour-code the Action column
    def _style_action(val):
        color = DECISION_COLORS.get(val, "#888888")
        return f"color: {color}; font-weight: bold;"

    # Use map() for pandas 2.1+, fall back to applymap() for older versions
    try:
        styled = (
            df.style
            .map(_style_action, subset=["Action"])
            .format({"Speed": "{:.2f}", "Conf": "{:.2f}", "A.Conf": "{:.2f}"})
        )
    except AttributeError:
        # Fallback for older pandas versions
        styled = (
            df.style
            .applymap(_style_action, subset=["Action"])
            .format({"Speed": "{:.2f}", "Conf": "{:.2f}", "A.Conf": "{:.2f}"})
        )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary badges
    counts: Dict[str, int] = {}
    for d in decisions:
        a = d.get("action", "MONITOR")
        counts[a] = counts.get(a, 0) + 1

    cols = st.columns(len(DECISION_COLORS))
    for i, (act, col_hex) in enumerate(DECISION_COLORS.items()):
        with cols[i]:
            n = counts.get(act, 0)
            st.metric(label=act, value=n)
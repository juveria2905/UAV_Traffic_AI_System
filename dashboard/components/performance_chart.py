"""
dashboard/components/performance_chart.py
Rolling FPS, latency, and decision-count charts.
Maintains its own session-state history buffer.
"""

import streamlit as st
import pandas as pd
from collections import deque
from typing import Deque
import time


_MAX_HISTORY = 200


def _init_history(key: str, maxlen: int = _MAX_HISTORY) -> deque:
    if key not in st.session_state:
        st.session_state[key] = deque(maxlen=maxlen)
    return st.session_state[key]


def render(snapshot: dict) -> None:
    """Render rolling performance charts."""
    fps          = snapshot.get("fps", 0.0)
    system_state = snapshot.get("system_state", {})

    if not system_state:
        st.info("System state not available yet.")
        return

    latency_ms   = system_state.get("agent_latency_ms", 0.0)
    emergency    = system_state.get("emergency_stops", 0)
    reroutes     = system_state.get("reroutes", 0)
    holds        = system_state.get("holds", 0)
    monitors     = system_state.get("monitors", 0)
    frame_idx    = snapshot.get("frame_idx", 0)
    ts           = time.time()

    # Append to rolling history
    fps_hist:     Deque = _init_history("perf_fps")
    lat_hist:     Deque = _init_history("perf_lat")
    dec_hist:     Deque = _init_history("perf_dec")

    fps_hist.append({"t": frame_idx, "FPS": fps})
    lat_hist.append({"t": frame_idx, "Latency (ms)": latency_ms})
    dec_hist.append({
        "t":              frame_idx,
        "EMERGENCY_STOP": emergency,
        "REROUTE":        reroutes,
        "HOLD":           holds,
        "MONITOR":        monitors,
    })

    # ── Metrics row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("FPS",             f"{fps:.1f}")
    c2.metric("Agent latency",   f"{latency_ms:.1f}ms")
    c3.metric("EMERGENCY_STOP",  emergency)
    c4.metric("REROUTE",         reroutes)
    c5.metric("HOLD",            holds)

    # ── FPS chart ──────────────────────────────────────────────────────────────
    fps_df = pd.DataFrame(list(fps_hist)).set_index("t")
    if not fps_df.empty:
        st.line_chart(fps_df, use_container_width=True, height=120)

    # ── Latency chart ──────────────────────────────────────────────────────────
    lat_df = pd.DataFrame(list(lat_hist)).set_index("t")
    if not lat_df.empty:
        st.caption("Agent pipeline latency (ms)")
        st.line_chart(lat_df, use_container_width=True, height=100)

    # ── Decision counts chart ──────────────────────────────────────────────────
    dec_df = pd.DataFrame(list(dec_hist)).set_index("t")
    if not dec_df.empty:
        st.caption("Decision counts per frame")
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            colors = {
                "EMERGENCY_STOP": "#ff3333",
                "REROUTE":        "#ff8800",
                "HOLD":           "#ffcc00",
                "MONITOR":        "#44cc44",
            }
            for col, color in colors.items():
                if col in dec_df.columns:
                    fig.add_trace(go.Scatter(
                        x=dec_df.index.tolist(),
                        y=dec_df[col].tolist(),
                        name=col,
                        line=dict(color=color, width=1.5),
                        mode="lines",
                    ))
            fig.update_layout(
                height=160,
                margin=dict(l=40, r=20, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.03)",
                showlegend=True,
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.area_chart(dec_df, use_container_width=True, height=120)

    # ── Feedback stats ─────────────────────────────────────────────────────────
    feedback_stats: dict = system_state.get("feedback_stats", {})
    if feedback_stats:
        with st.expander("Feedback engine stats", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Evaluated",     feedback_stats.get("total_evaluated", 0))
            c2.metric("Pending",       feedback_stats.get("pending_count", 0))
            c3.metric("Avg reward",    f"{feedback_stats.get('overall_avg_reward', 0):.3f}")

            per_action = feedback_stats.get("per_action", {})
            if per_action:
                rows = [
                    {"Action": a, "Count": v.get("count", 0),
                     "Avg reward": round(v.get("avg_reward", 0), 3)}
                    for a, v in per_action.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
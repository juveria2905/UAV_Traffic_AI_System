"""
================================================================================
dashboard/app.py — UAV Traffic AI Streamlit Dashboard
================================================================================

Run:
    streamlit run dashboard/app.py

The dashboard polls the FastAPI backend at http://localhost:8000 via:
  GET /api/snapshot — full system snapshot (no frame image)
  GET /api/frame    — latest annotated frame as base64 JPEG

All data is sourced exclusively from SharedPipelineState (backend/api.py)
which is populated by main.py on every processed frame.

Tabs:
  Live Feed        — annotated frame + collision alerts + active tracks
  Agent Intelligence — goals, confidence, hierarchy, communication
  Analytics        — heatmaps, hotspots, speed distribution
  Reasoning        — per-decision reasoning chains
  Memory           — DecisionMemory stats + feedback engine
  Performance      — FPS, latency, decision throughput
================================================================================
"""

import sys
import time
from pathlib import Path

import requests
import streamlit as st

# Ensure project root is on path when running from dashboard/
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import API_HOST, API_PORT, DASHBOARD_CONFIG, DECISION_COLORS_HEX

from dashboard.tabs import (
    live_feed,
    agent_intelligence,
    analytics,
    reasoning,
    memory,
    performance,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_CONFIG.title,
    page_icon=DASHBOARD_CONFIG.page_icon,
    layout=DASHBOARD_CONFIG.layout,
    initial_sidebar_state="expanded",
)

API_BASE = f"http://{API_HOST}:{API_PORT}"


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=0.2)
def _check_api_health() -> bool:
    """Check if the API is available. Cached for 200ms."""
    try:
        resp = requests.get(f"{API_BASE}/", timeout=1.0)
        return resp.ok
    except Exception:
        return False


@st.cache_data(ttl=0.1)
def _fetch_snapshot() -> dict:
    """Fetch current system snapshot from API. Cached for 100ms."""
    try:
        resp = requests.get(f"{API_BASE}/api/snapshot", timeout=1.0)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return {}


def _status_badge(is_running: bool, fps: float) -> str:
    if is_running:
        return f"🟢 **RUNNING** &nbsp; FPS: {fps:.1f}"
    return "🔴 **STOPPED**"


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar(snapshot: dict) -> None:
    with st.sidebar:
        st.title("🚁 UAV Traffic AI")
        st.caption("Hierarchical Agentic System")

        st.divider()

        is_run    = snapshot.get("is_running", False)
        fps       = snapshot.get("fps", 0.0)
        frame_idx = snapshot.get("frame_idx", 0)

        st.markdown(_status_badge(is_run, fps), unsafe_allow_html=True)
        st.metric("Frame",          frame_idx)
        st.metric("Active tracks",  len(snapshot.get("tracks", [])))
        st.metric("Collisions",     len(snapshot.get("collision_events", [])))

        st.divider()

        # Decision counts
        system_state = snapshot.get("system_state", {})
        st.subheader("Agent Decisions")
        for action, color in DECISION_COLORS_HEX.items():
            key = {
                "EMERGENCY_STOP": "emergency_stops",
                "REROUTE":        "reroutes",
                "HOLD":           "holds",
                "MONITOR":        "monitors",
            }.get(action, action.lower() + "s")
            count = system_state.get(key, 0)
            st.markdown(
                f"<span style='color:{color};font-weight:bold'>{action}</span>: {count}",
                unsafe_allow_html=True,
            )

        st.divider()

        # Goal summary
        goal_summary = system_state.get("goal_summary", {})
        if goal_summary:
            st.subheader("Dominant Goal")
            dominant = goal_summary.get("dominant_goal", "NONE")
            n_goals  = goal_summary.get("active_goal_count", 0)
            st.info(f"**{dominant}**\n\n{n_goals} active goal(s)")

        st.divider()

        # Refresh control
        refresh_ms = st.slider(
            "Refresh interval (ms)",
            min_value=100,
            max_value=2000,
            value=DASHBOARD_CONFIG.refresh_rate_ms,
            step=100,
            key="refresh_ms",
        )

        st.caption(f"API: {API_BASE}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not _check_api_health():
        st.error(
            f"❌ Cannot reach API at {API_BASE}\n\n"
            f"Make sure to run: `python main.py` in another terminal"
        )
        st.stop()

    snapshot = _fetch_snapshot()

    _render_sidebar(snapshot)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📹 Live Feed",
        "🤖 Agent Intelligence",
        "📊 Analytics",
        "🧠 Reasoning",
        "💾 Memory",
        "⚡ Performance",
    ])

    with tabs[0]:
        live_feed.render(snapshot, API_BASE)

    with tabs[1]:
        agent_intelligence.render(snapshot, API_BASE)

    with tabs[2]:
        analytics.render(snapshot)

    with tabs[3]:
        reasoning.render(snapshot)

    with tabs[4]:
        memory.render(snapshot, API_BASE)

    with tabs[5]:
        performance.render(snapshot)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    refresh_ms = st.session_state.get("refresh_ms", DASHBOARD_CONFIG.refresh_rate_ms)
    time.sleep(refresh_ms / 1000.0)
    st.rerun()


if __name__ == "__main__":
    main()
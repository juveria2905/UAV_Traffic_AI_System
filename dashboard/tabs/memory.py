"""
dashboard/tabs/memory.py
Decision memory statistics and per-track history.
"""

import streamlit as st
import pandas as pd
import requests
from typing import List

from dashboard.components import memory_timeline


def render(snapshot: dict, api_base: str) -> None:
    """Render the decision memory tab."""
    st.subheader("Decision Memory System")
    st.caption(
        "The DecisionMemory stores each agent decision with outcome tracking. "
        "FeedbackEngine evaluates outcomes and writes reward scores back, "
        "enabling adaptive confidence scoring."
    )

    memory_timeline.render(snapshot)

    st.divider()

    # ── Per-track memory via API /memory endpoint ─────────────────────────────
    st.subheader("Per-Track Memory Summaries")
    try:
        resp = requests.get(f"{api_base}/memory", timeout=1.0)
        if resp.ok:
            data    = resp.json()
            recent  = data.get("recent", [])
            if recent and isinstance(recent, list):
                rows = []
                for r in recent:
                    if not isinstance(r, dict):
                        continue
                    rows.append({
                        "Track":        r.get("track_id", "?"),
                        "Action":       r.get("action", "?"),
                        "Confidence":   round(r.get("confidence", 0), 3),
                        "Outcome":      round(r.get("outcome_score", 0), 3),
                        "Evaluated":    r.get("evaluated", False),
                        "Frame":        r.get("frame_idx", "?"),
                        "Agent":        r.get("agent", "?"),
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No memory records via API yet.")
            else:
                st.info("No memory records via API yet.")
        else:
            st.warning(f"Memory API returned {resp.status_code}.")
    except requests.exceptions.Timeout:
        st.warning("Memory API timeout.")
    except requests.exceptions.RequestException:
        st.info("Memory API unavailable (pipeline may not expose memory endpoint).")
    except Exception as e:
        st.debug(f"Memory API error: {e}")

    # ── Feedback stats ─────────────────────────────────────────────────────────
    system_state   = snapshot.get("system_state", {})
    feedback_stats = system_state.get("feedback_stats", {})

    if not feedback_stats:
        st.info("Feedback engine stats not available yet.")
        return

    st.divider()
    st.subheader("Feedback Engine")
    per_action: dict = feedback_stats.get("per_action", {})
    if per_action and isinstance(per_action, dict):
        rows = [
            {
                "Action":     a,
                "Count":      v.get("count", 0) if isinstance(v, dict) else 0,
                "Avg reward": round(v.get("avg_reward", 0) if isinstance(v, dict) else 0, 4),
                "Total":      round(v.get("total", 0) if isinstance(v, dict) else 0, 3),
            }
            for a, v in per_action.items()
        ]
        if rows:
            df = pd.DataFrame(rows)

            if not df.empty and "Avg reward" in df.columns:
                def _color_reward(val):
                    if isinstance(val, (int, float)):
                        if val > 0.2:  return "color: #44cc44"
                        if val < 0:    return "color: #ff4444"
                    return ""

                styled = df.style.map(_color_reward, subset=["Avg reward"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No feedback data available yet.")
    else:
        st.info("No per-action feedback data available.")
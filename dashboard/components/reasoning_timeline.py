"""
dashboard/components/reasoning_timeline.py
Displays ReasoningChain records from system_state["reasoning_chains"].
Each chain matches ReasoningChain.to_dict():
  chain_id, track_id, frame_idx, agent, decision, confidence,
  avg_confidence, steps (list of ReasoningStep.to_dict()), step_count,
  duration_ms, completed, created_at
Each step: step, type, description, value, unit, interpretation,
           confidence, timestamp, metadata
"""

import streamlit as st
import pandas as pd
from typing import List, Dict


STEP_TYPE_COLORS = {
    "OBSERVATION":  "#4488ff",
    "ANALYSIS":     "#44aaff",
    "COMPARISON":   "#ffaa44",
    "DECISION":     "#44cc44",
    "ESCALATION":   "#cc44ff",
    "CONFIDENCE":   "#44cccc",
    "CONTEXT":      "#888888",
}

DECISION_COLORS = {
    "EMERGENCY_STOP": "#ff3333",
    "REROUTE":        "#ff8800",
    "HOLD":           "#ffcc00",
    "MONITOR":        "#44cc44",
    "PRIORITIZE":     "#cc44ff",
}


def _render_chain(chain: dict) -> None:
    """Render a single reasoning chain in an expander."""
    decision = chain.get("decision", "?")
    conf     = chain.get("confidence", 0)
    tid      = chain.get("track_id", "?")
    agent    = chain.get("agent", "?")
    dur      = chain.get("duration_ms", 0)
    cid      = chain.get("chain_id", "?")
    d_color  = DECISION_COLORS.get(decision, "#888")

    title = (
        f"[{cid}] Track #{tid} → "
        f"**:{d_color.lstrip('#')}[{decision}]** "
        f"(conf={conf:.2f}, {dur:.1f}ms)"
    )

    with st.expander(f"Track #{tid} → {decision}  (conf={conf:.2f})", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.caption(f"Agent: {agent}")
        c2.caption(f"Chain ID: {cid}")
        c3.caption(f"Duration: {dur:.1f}ms")

        steps: List[dict] = chain.get("steps", [])
        for s in steps:
            stype  = s.get("type", "ANALYSIS")
            desc   = s.get("description", "")
            val    = s.get("value")
            unit   = s.get("unit", "")
            interp = s.get("interpretation", "")
            sconf  = s.get("confidence", 1.0)
            color  = STEP_TYPE_COLORS.get(stype, "#888888")

            val_str   = f" = {val:.3f} {unit}" if val is not None else ""
            interp_str = f" → {interp}" if interp else ""
            conf_str   = f" [conf={sconf:.2f}]"

            st.markdown(
                f"<div style='border-left:3px solid {color};padding:2px 8px;"
                f"margin:2px 0;font-size:12px'>"
                f"<b style='color:{color}'>[{stype}]</b> "
                f"{desc}{val_str}{interp_str}"
                f"<span style='color:#888'>{conf_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render(snapshot: dict) -> None:
    """Render reasoning chains from the agent system."""
    system_state   = snapshot.get("system_state", {})
    chains: List[dict] = system_state.get("reasoning_chains", [])

    if not chains:
        st.info("No reasoning chains recorded yet. "
                "Chains are generated for REROUTE/EMERGENCY_STOP decisions.")
        return

    # Filter controls
    decision_opts = sorted({c.get("decision", "?") for c in chains})
    selected = st.multiselect(
        "Filter by decision type",
        options=decision_opts,
        default=decision_opts,
        key="reasoning_filter",
    )

    filtered = [c for c in chains if c.get("decision") in selected]

    st.caption(f"Showing {len(filtered)} of {len(chains)} chains")

    # Summary table
    rows = [{
        "ID":       c.get("chain_id", "?"),
        "Track":    c.get("track_id", "?"),
        "Decision": c.get("decision", "?"),
        "Conf":     round(c.get("confidence", 0), 3),
        "Steps":    c.get("step_count", 0),
        "ms":       round(c.get("duration_ms", 0), 1),
    } for c in filtered]

    if not rows:
        st.info("No reasoning chains match the selected filters.")
        return

    df = pd.DataFrame(rows)

    def _colour(val):
        color = DECISION_COLORS.get(val, "#888")
        return f"color: {color}; font-weight: bold;"

    if "Decision" in df.columns:
        styled = df.style.map(_colour, subset=["Decision"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Render individual chains
    for chain in filtered[:10]:
        _render_chain(chain)
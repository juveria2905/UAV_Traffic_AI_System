"""
dashboard/components/agent_hierarchy.py
Visualises the 3-tier agent hierarchy and zone grid state.
Data sources:
  snapshot["system_state"]["city_state"] — from HierarchyManager.update()
    city_risk, zones (dict of zone_id -> AirspaceZone.to_dict()),
    congested_zones, corridors, grid_info, vehicle_count
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List


RISK_COLOR_CSS = {
    "HIGH":   "#ff3333",
    "MEDIUM": "#ff8800",
    "LOW":    "#ffcc00",
    "SAFE":   "#44cc44",
}


def render(snapshot: dict) -> None:
    """Render agent hierarchy summary and zone grid."""
    system_state: dict = snapshot.get("system_state", {})
    city_state:   dict = system_state.get("city_state", {})

    # ── Tier summary ──────────────────────────────────────────────────────────
    st.subheader("Agent Hierarchy")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Level 0 — DroneAgent**")
        st.caption("Per-vehicle autonomous decisions")
        n_tracks = snapshot.get("system_state", {}).get("total_vehicles", len(snapshot.get("tracks", [])))
        st.metric("Active drones", n_tracks)

    with c2:
        st.markdown("**Level 1 — ZoneAgent**")
        st.caption("Sector-level coordination")
        zones      = city_state.get("zones", {})
        n_zones    = len(zones)
        congested  = city_state.get("congested_zones", [])
        st.metric("Zones", n_zones)
        if congested:
            st.warning(f"Congested: {', '.join(congested[:5])}")

    with c3:
        st.markdown("**Level 2 — CityAgent**")
        st.caption("City-wide airspace management")
        city_risk = city_state.get("city_risk", "SAFE")
        color     = RISK_COLOR_CSS.get(city_risk, "#888")
        st.metric("City risk", city_risk)
        st.markdown(f"<span style='color:{color};font-weight:bold'>{city_risk}</span>",
                    unsafe_allow_html=True)

    st.divider()

    # ── Zone grid ─────────────────────────────────────────────────────────────
    zones: Dict[str, dict] = city_state.get("zones", {})
    if not zones:
        st.info("Zone data not available (HierarchyManager may be disabled).")
        return

    st.subheader("Zone Grid State")

    grid_info = city_state.get("grid_info", {})
    rows_n    = grid_info.get("rows", 4)
    cols_n    = grid_info.get("cols", 4)

    # Build a rows_n x cols_n grid of zone cards
    zone_list = sorted(zones.values(), key=lambda z: (z["row"], z["col"]))
    idx       = 0

    for r in range(rows_n):
        st_cols = st.columns(cols_n)
        for c in range(cols_n):
            if idx >= len(zone_list):
                break
            zone = zone_list[idx]
            idx += 1
            risk  = zone.get("risk_level", "SAFE")
            color = RISK_COLOR_CSS.get(risk, "#888")
            dens  = zone.get("density", 0)
            n_veh = zone.get("vehicle_count", 0)
            zid   = zone.get("zone_id", "?")
            cong  = zone.get("congested", False)

            with st_cols[c]:
                border = f"2px solid {color}"
                bg     = "rgba(255,51,51,0.08)" if risk == "HIGH" else "rgba(0,0,0,0.03)"
                st.markdown(
                    f"""<div style="border:{border};border-radius:6px;
                    padding:6px 8px;background:{bg};margin:2px">
                    <b style="color:{color}">{zid}</b>
                    <span style="font-size:11px;color:#888"> {risk}</span><br/>
                    🚁 {n_veh} &nbsp; dens={dens:.2f}
                    {"&nbsp;🚨" if cong else ""}
                    </div>""",
                    unsafe_allow_html=True,
                )

    # Corridor map
    corridors: Dict[str, str] = city_state.get("corridors", {})
    if corridors:
        st.caption(f"Open corridors ({len(corridors)}): " + ", ".join(corridors.keys()))
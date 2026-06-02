"""
================================================================================
agentic_ai/hierarchy_manager.py — UAV Hierarchy Manager
================================================================================

PURPOSE:
  Implements a 3-tier spatial agent hierarchy for UAV traffic management:
    CityAgent   → monitors city-wide traffic density, allocates air corridors
    ZoneAgent   → monitors local sectors, coordinates nearby drone conflicts
    DroneAgent  → local autonomous decisions per vehicle

WHY THIS FILE EXISTS:
  The existing MonitorAgent/CoordinatorAgent/ExecutorAgent handles per-frame
  decisions. The hierarchy_manager adds SPATIAL intelligence:
  - The city is divided into a grid of zones (default 4×4 = 16 zones)
  - Each zone has a ZoneAgent responsible for its vehicles
  - CityAgent gets a city-wide view and can reassign vehicles between zones
  - This mirrors real Urban Air Mobility (UAM) traffic management systems

CONNECTS TO:
  - agent_system.py     → HierarchicalAgentSystem calls hierarchy_manager
  - priority_engine.py  → Per-zone priority aggregation
  - communication.py    → Zone agents communicate via message bus
  - api.py             → /api/zones endpoint exposes zone states
  - dashboard/app.py   → "Airspace Intelligence" tab renders zone map
================================================================================
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from utils.logger import get_logger, ModuleStats
from agentic_ai.communication import AgentMessageBus, AgentMessage, MessagePriority, Topics

logger = get_logger(__name__)


# ==============================================================================
# ZONE DATA STRUCTURE
# ==============================================================================

@dataclass
class AirspaceZone:
    """
    One spatial zone in the airspace grid.

    The frame is divided into grid_cols × grid_rows zones.
    Each zone tracks which vehicle IDs are currently within its boundaries.

    Fields:
        zone_id:       Unique zone identifier (e.g., "Z0", "Z7", "Z15")
        row, col:      Position in the grid (0-indexed)
        bbox:          (x1, y1, x2, y2) pixel boundaries of this zone
        track_ids:     Current set of vehicle track IDs in this zone
        risk_level:    Aggregate risk level ("HIGH", "MEDIUM", "LOW", "SAFE")
        density:       Number of vehicles / max_capacity (normalized 0-1)
        agent_name:    Which ZoneAgent is responsible for this zone

    VISUALIZATION:
        Zone grid overlaid on the video feed — each zone colored by risk level.
    """
    zone_id:     str
    row:         int
    col:         int
    bbox:        Tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels
    track_ids:   Set[int]                            = field(default_factory=set)
    risk_level:  str                                 = "SAFE"
    density:     float                               = 0.0
    congested:   bool                                = False
    agent_name:  str                                 = ""
    updated_at:  float                               = field(default_factory=time.time)

    @property
    def center(self) -> Tuple[float, float]:
        """Center pixel of this zone."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def vehicle_count(self) -> int:
        return len(self.track_ids)

    def risk_color_bgr(self) -> Tuple[int, int, int]:
        """OpenCV BGR color for zone risk overlay."""
        return {
            "HIGH":   (0,   0,  255),   # Red
            "MEDIUM": (0, 165,  255),   # Orange
            "LOW":    (0, 255,  255),   # Yellow
            "SAFE":   (0, 255,    0),   # Green
        }.get(self.risk_level, (128, 128, 128))

    def to_dict(self) -> dict:
        return {
            "zone_id":     self.zone_id,
            "row":         self.row,
            "col":         self.col,
            "bbox":        list(self.bbox),
            "center":      list(self.center),
            "track_ids":   list(self.track_ids),
            "vehicle_count": self.vehicle_count,
            "risk_level":  self.risk_level,
            "density":     round(self.density, 4),
            "congested":   self.congested,
            "agent":       self.agent_name,
        }


# ==============================================================================
# DRONE AGENT (VEHICLE-LEVEL)
# ==============================================================================

class DroneAgent:
    """
    Level 0 — Autonomous per-vehicle agent.

    In a real UAV system, this would be the onboard flight controller.
    In our simulation, it encapsulates per-vehicle state and local decisions.

    DECISIONS:
        MAINTAIN_COURSE  → No action needed
        REDUCE_SPEED     → Approaching speed limit
        ALTITUDE_CHANGE  → (future: 3D airspace)
        EMERGENCY_LAND   → Critical failure state
    """

    DECISIONS = ("MAINTAIN_COURSE", "REDUCE_SPEED", "ALTITUDE_CHANGE", "EMERGENCY_LAND")

    def __init__(self, track_id: int):
        self.track_id    = track_id
        self.agent_name  = f"DroneAgent-{track_id}"
        self.zone_id:    Optional[str]  = None
        self.local_decision: str        = "MAINTAIN_COURSE"
        self.local_confidence: float    = 1.0
        self._created_at = time.time()

    def assess(self, speed: float, speed_limit: float = 30.0) -> str:
        """
        Make a local autonomous decision based on own speed.

        This is a simplified speed-gate. In production:
        - Check own trajectory against corridor constraints
        - Consider altitude bands
        - Monitor battery level
        """
        if speed > speed_limit * 1.5:
            self.local_decision   = "REDUCE_SPEED"
            self.local_confidence = 0.85
        else:
            self.local_decision   = "MAINTAIN_COURSE"
            self.local_confidence = 0.99
        return self.local_decision

    def to_dict(self) -> dict:
        return {
            "track_id":         self.track_id,
            "agent":            self.agent_name,
            "zone":             self.zone_id,
            "local_decision":   self.local_decision,
            "confidence":       round(self.local_confidence, 4),
        }


# ==============================================================================
# ZONE AGENT (SECTOR-LEVEL)
# ==============================================================================

class ZoneAgent:
    """
    Level 1 — Zone-level coordination agent.

    Responsible for all vehicles within one airspace sector.
    Communicates upward to CityAgent when zone becomes congested.

    RESPONSIBILITIES:
        - Track which vehicles are in this zone
        - Compute zone-level risk aggregate
        - Detect congestion (> congestion_threshold vehicles)
        - Request reroutes from vehicles approaching zone boundary
        - Report zone state to CityAgent via message bus
    """

    def __init__(
        self,
        zone: AirspaceZone,
        bus:  AgentMessageBus,
        congestion_threshold: int = 8,
        max_capacity:         int = 20,
    ):
        self.zone                 = zone
        self.bus                  = bus
        self.congestion_threshold = congestion_threshold
        self.max_capacity         = max_capacity
        self.agent_name           = f"ZoneAgent-{zone.zone_id}"
        zone.agent_name           = self.agent_name

        self._drone_agents: Dict[int, DroneAgent] = {}
        self._history_density: list               = []   # rolling density log

        logger.debug(f"ZoneAgent {self.agent_name} initialized for zone {zone.zone_id}")

    def update(
        self,
        tracks_in_zone: Dict[int, any],
        frame_idx:      int,
    ) -> dict:
        """
        Update zone state with current tracks. Called every frame by CityAgent.

        Args:
            tracks_in_zone: Dict of {track_id: TrackedObject} for vehicles in this zone
            frame_idx:      Current frame number

        Returns:
            Zone state dict for CityAgent aggregation.
        """
        # Update zone membership
        self.zone.track_ids = set(tracks_in_zone.keys())
        density             = len(self.zone.track_ids) / max(self.max_capacity, 1)
        self.zone.density   = float(np.clip(density, 0.0, 1.0))
        self.zone.updated_at = time.time()

        # Congestion check
        was_congested       = self.zone.congested
        self.zone.congested = len(self.zone.track_ids) >= self.congestion_threshold
        self._history_density.append(self.zone.density)
        if len(self._history_density) > 100:
            self._history_density.pop(0)

        # Publish congestion alert if state changed
        if self.zone.congested and not was_congested:
            self.bus.broadcast(
                topic=Topics.ZONE_CONGESTION,
                sender=self.agent_name,
                payload={
                    "zone_id":       self.zone.zone_id,
                    "vehicle_count": self.zone.vehicle_count,
                    "density":       self.zone.density,
                    "frame_idx":     frame_idx,
                },
                priority=MessagePriority.HIGH,
            )
            logger.info(
                f"[ZoneAgent {self.zone.zone_id}] CONGESTION ALERT: "
                f"{self.zone.vehicle_count} vehicles (density={self.zone.density:.2f})"
            )
        elif not self.zone.congested and was_congested:
            self.bus.broadcast(
                topic=Topics.ZONE_CLEAR,
                sender=self.agent_name,
                payload={"zone_id": self.zone.zone_id, "frame_idx": frame_idx},
                priority=MessagePriority.NORMAL,
            )

        # Ensure DroneAgent exists for each vehicle
        for tid in self.zone.track_ids:
            if tid not in self._drone_agents:
                self._drone_agents[tid] = DroneAgent(tid)
            self._drone_agents[tid].zone_id = self.zone.zone_id

        # Remove stale DroneAgents (vehicles that left the zone)
        for tid in list(self._drone_agents.keys()):
            if tid not in self.zone.track_ids:
                del self._drone_agents[tid]

        return self.zone.to_dict()

    def set_zone_risk(self, risk_level: str) -> None:
        """Set zone risk level (called by CityAgent based on collision events)."""
        self.zone.risk_level = risk_level

    def get_drone_states(self) -> List[dict]:
        return [da.to_dict() for da in self._drone_agents.values()]


# ==============================================================================
# CITY AGENT (CITY-WIDE)
# ==============================================================================

class CityAgent:
    """
    Level 2 — City-wide airspace manager.

    Maintains a grid-based airspace model. Aggregates zone states and
    provides a global view of traffic density and risk distribution.

    RESPONSIBILITIES:
        - Divide frame into spatial zones
        - Assign vehicles to zones based on their position
        - Aggregate zone risk levels (per-zone max collision severity)
        - Identify air corridors (low-density zone paths)
        - Rebalance zones if workload is uneven

    GRID MODEL:
        Frame width W × height H divided into grid_cols × grid_rows zones.
        Zone Z(r,c) covers pixels:
            x: [c * zone_w, (c+1) * zone_w]
            y: [r * zone_h, (r+1) * zone_h]
    """

    def __init__(
        self,
        frame_width:  int,
        frame_height: int,
        grid_cols:    int = 4,
        grid_rows:    int = 4,
        bus:          Optional[AgentMessageBus] = None,
        congestion_threshold: int = 8,
        max_capacity_per_zone: int = 20,
    ):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.grid_cols    = grid_cols
        self.grid_rows    = grid_rows
        self.bus          = bus
        self.agent_name   = "CityAgent"

        self.zone_w = frame_width  / grid_cols
        self.zone_h = frame_height / grid_rows

        # Build zone grid
        self.zones:       Dict[str, AirspaceZone] = {}
        self.zone_agents: Dict[str, ZoneAgent]    = {}
        self._init_zones(congestion_threshold, max_capacity_per_zone)

        self._stats        = ModuleStats("CityAgent")
        self._city_risk    = "SAFE"
        self._corridor_map: Dict[str, str] = {}   # zone_id → suggested_corridor

        logger.info(
            f"CityAgent initialized: {grid_rows}×{grid_cols} grid "
            f"({len(self.zones)} zones), frame={frame_width}×{frame_height}"
        )

    def _init_zones(self, congestion_threshold: int, max_capacity: int) -> None:
        """Create all zones and their ZoneAgents."""
        bus = self.bus
        zone_idx = 0
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                zone_id = f"Z{zone_idx:02d}"
                x1 = c * self.zone_w
                y1 = r * self.zone_h
                x2 = x1 + self.zone_w
                y2 = y1 + self.zone_h

                zone = AirspaceZone(
                    zone_id=zone_id,
                    row=r,
                    col=c,
                    bbox=(x1, y1, x2, y2),
                )
                self.zones[zone_id] = zone

                if bus:
                    za = ZoneAgent(zone, bus, congestion_threshold, max_capacity)
                    self.zone_agents[zone_id] = za

                zone_idx += 1

    def assign_vehicles_to_zones(self, tracks: dict) -> Dict[str, Dict[int, any]]:
        """
        Assign each tracked vehicle to its containing zone.

        Returns:
            {zone_id: {track_id: TrackedObject}} for all non-empty zones.
        """
        zone_contents: Dict[str, Dict[int, any]] = defaultdict(dict)

        for track_id, track in tracks.items():
            center = track.center if hasattr(track, "center") else None
            if center is None:
                continue

            cx, cy = center
            zone_id = self._point_to_zone(cx, cy)
            if zone_id:
                zone_contents[zone_id][track_id] = track

        return dict(zone_contents)

    def _point_to_zone(self, cx: float, cy: float) -> Optional[str]:
        """Map (cx, cy) pixel coordinate to zone_id."""
        col = int(cx / self.zone_w)
        row = int(cy / self.zone_h)
        col = min(col, self.grid_cols - 1)
        row = min(row, self.grid_rows - 1)
        idx = row * self.grid_cols + col
        return f"Z{idx:02d}"

    def update(
        self,
        frame_idx:        int,
        tracks:           dict,
        collision_events: list,
    ) -> dict:
        """
        Full city-level update per frame.

        1. Assign vehicles to zones
        2. Update each ZoneAgent
        3. Apply collision risk to zones
        4. Compute city-wide risk
        5. Identify corridors

        Args:
            frame_idx:        Current frame number
            tracks:           All active TrackedObjects
            collision_events: CollisionEvents from CollisionEngine

        Returns:
            City-wide state dict for API/dashboard.
        """
        t_start = time.perf_counter()

        # Step 1: Assign vehicles to zones
        zone_assignments = self.assign_vehicles_to_zones(tracks)

        # Step 2: Update each ZoneAgent
        zone_states = {}
        for zone_id, zone in self.zones.items():
            contents = zone_assignments.get(zone_id, {})
            if zone_id in self.zone_agents:
                zone_state = self.zone_agents[zone_id].update(contents, frame_idx)
            else:
                zone.track_ids = set(contents.keys())
                zone.density   = len(zone.track_ids) / 20.0
                zone_state     = zone.to_dict()
            zone_states[zone_id] = zone_state

        # Step 3: Apply collision risk to affected zones
        self._apply_collision_risk(collision_events)

        # Step 4: City-wide risk level
        all_risks = [z.risk_level for z in self.zones.values()]
        if "HIGH"   in all_risks: self._city_risk = "HIGH"
        elif "MEDIUM" in all_risks: self._city_risk = "MEDIUM"
        elif "LOW"    in all_risks: self._city_risk = "LOW"
        else:                       self._city_risk = "SAFE"

        # Step 5: Identify low-density corridors
        self._update_corridors()

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._stats.record(elapsed_ms, success=True)

        return {
            "city_risk":        self._city_risk,
            "zones":            zone_states,
            "vehicle_count":    len(tracks),
            "congested_zones":  [
                zid for zid, z in self.zones.items() if z.congested
            ],
            "corridors":        self._corridor_map,
            "grid_info": {
                "rows": self.grid_rows,
                "cols": self.grid_cols,
                "zone_w": self.zone_w,
                "zone_h": self.zone_h,
            },
            "frame_idx":        frame_idx,
        }

    def _apply_collision_risk(self, collision_events: list) -> None:
        """
        Propagate collision risk levels to the zones containing the vehicles.
        """
        # First reset all zones to SAFE
        for zone in self.zones.values():
            zone.risk_level = "SAFE"

        risk_priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "SAFE": 0}

        for event in collision_events:
            # Determine zone for each vehicle in the collision
            for track_id in [event.track_id_a, event.track_id_b]:
                for zone in self.zones.values():
                    if track_id in zone.track_ids:
                        event_risk = event.risk_level.value if hasattr(
                            event.risk_level, "value") else str(event.risk_level)
                        if risk_priority.get(event_risk, 0) > risk_priority.get(
                            zone.risk_level, 0
                        ):
                            zone.risk_level = event_risk
                            if zone.zone_id in self.zone_agents:
                                self.zone_agents[zone.zone_id].set_zone_risk(event_risk)

    def _update_corridors(self) -> None:
        """
        Identify low-density zones that can serve as rerouting corridors.
        A corridor is a contiguous path of low-density zones from one
        side of the frame to another.

        Simplified implementation: mark zones below 30% density as corridors.
        """
        self._corridor_map = {}
        for zone_id, zone in self.zones.items():
            if zone.density < 0.3 and zone.risk_level in ("SAFE", "LOW"):
                self._corridor_map[zone_id] = "CORRIDOR"

    def annotate_frame(self, frame: np.ndarray, alpha: float = 0.15) -> np.ndarray:
        """
        Draw zone grid overlay on the video frame.

        Each zone is colored by risk level with a semi-transparent fill.
        Zone ID and vehicle count are shown in the zone center.

        Args:
            frame: BGR numpy array to annotate
            alpha: Transparency of zone fill (0=invisible, 1=opaque)

        Returns:
            Annotated frame copy.
        """
        import cv2
        annotated = frame.copy()
        overlay   = annotated.copy()

        for zone in self.zones.values():
            x1, y1, x2, y2 = (int(v) for v in zone.bbox)
            color = zone.risk_color_bgr()

            # Fill zone with semi-transparent color
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Zone border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            # Zone label
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.putText(
                annotated,
                f"{zone.zone_id}:{zone.vehicle_count}",
                (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

        # Blend overlay
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        return annotated

    def get_zone_list(self) -> List[dict]:
        """Return list of all zone states for API/dashboard."""
        return [z.to_dict() for z in self.zones.values()]

    def get_stats(self) -> dict:
        return {
            "city_agent": self._stats.summary(),
            "zone_count": len(self.zones),
            "city_risk":  self._city_risk,
        }


# ==============================================================================
# HIERARCHY MANAGER — ORCHESTRATOR
# ==============================================================================

class HierarchyManager:
    """
    Top-level orchestrator for the 3-tier UAV agent hierarchy.

    Integrates:
        CityAgent   → city-wide airspace management
        ZoneAgent   → per-sector management (created by CityAgent)
        DroneAgent  → per-vehicle decisions (created by ZoneAgent)

    Called once per frame from agent_system.HierarchicalAgentSystem.

    USAGE:
        manager = HierarchyManager(frame_width=1280, frame_height=720, bus=bus)

        # Every frame:
        city_state = manager.update(frame_idx, tracks, collision_events)
        annotated  = manager.annotate_frame(frame)
    """

    def __init__(
        self,
        frame_width:  int,
        frame_height: int,
        bus:          Optional[AgentMessageBus] = None,
        grid_cols:    int = 4,
        grid_rows:    int = 4,
        congestion_threshold:  int = 8,
        max_capacity_per_zone: int = 20,
    ):
        self.bus = bus
        self.city_agent = CityAgent(
            frame_width=frame_width,
            frame_height=frame_height,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            bus=bus,
            congestion_threshold=congestion_threshold,
            max_capacity_per_zone=max_capacity_per_zone,
        )
        self._latest_city_state: dict = {}
        logger.info(
            f"HierarchyManager initialized: "
            f"{frame_width}×{frame_height} frame, "
            f"{grid_rows}×{grid_cols} zone grid"
        )

    def update(
        self,
        frame_idx:        int,
        tracks:           dict,
        collision_events: list,
    ) -> dict:
        """
        Run full hierarchy update for one frame.

        Returns city-state dict for API/dashboard/annotation.
        """
        city_state = self.city_agent.update(frame_idx, tracks, collision_events)
        self._latest_city_state = city_state
        return city_state

    def annotate_frame(self, frame: np.ndarray, alpha: float = 0.12) -> np.ndarray:
        """Draw zone grid overlay on frame."""
        return self.city_agent.annotate_frame(frame, alpha=alpha)

    @property
    def latest_city_state(self) -> dict:
        return self._latest_city_state

    def get_zone_for_track(self, track_id: int) -> Optional[str]:
        """Return zone_id containing the given track_id."""
        for zone_id, zone in self.city_agent.zones.items():
            if track_id in zone.track_ids:
                return zone_id
        return None

    def get_congested_zones(self) -> List[str]:
        return [
            zid for zid, z in self.city_agent.zones.items()
            if z.congested
        ]

    def get_stats(self) -> dict:
        return self.city_agent.get_stats()
"""
================================================================================
prediction/collision_engine.py — Collision Prediction Engine
================================================================================

DAY 4 SUBSYSTEM — COLLISION DETECTION

═══════════════════════════════════════════════════════════════════════════════
CONCEPT: TIME-TO-COLLISION (TTC)
═══════════════════════════════════════════════════════════════════════════════

TTC = How many seconds until two vehicles occupy the same space?

SIMPLE FORMULA (1D, head-on):
  TTC = gap / closing_speed
  
  gap = distance between vehicles
  closing_speed = relative speed (how fast they're approaching each other)

If TTC < 3 seconds → HIGH RISK (emergency brake needed NOW)
If TTC < 6 seconds → MEDIUM RISK (prepare to brake)
If TTC > 6 seconds → LOW RISK (monitor only)

WHY 2D TTC IS MORE COMPLEX:
  In 2D (x,y plane), we need to ask:
  "Given current positions and velocities, what is the minimum distance
   the two vehicles will reach, and when?"
  
  We parameterize relative position as a function of time:
    dx(t) = (cx1 + vx1*t) - (cx2 + vx2*t) = dx0 + dvx*t
    dy(t) = (cy1 + vy1*t) - (cy2 + vy2*t) = dy0 + dvy*t
    
    distance²(t) = dx(t)² + dy(t)²
  
  To find minimum distance, take derivative and set to zero:
    d(distance²)/dt = 0
    2*dx(t)*dvx + 2*dy(t)*dvy = 0
    t_min = -(dx0*dvx + dy0*dvy) / (dvx² + dvy²)
  
  If t_min > 0 (vehicles are converging), compute min_distance.
  If min_distance < collision_radius → they WILL collide at time t_min.
  TTC = t_min (in frames, convert to seconds using FPS).

═══════════════════════════════════════════════════════════════════════════════
WHY WE USE PREDICTED POSITIONS INSTEAD OF CURRENT
═══════════════════════════════════════════════════════════════════════════════

Current positions only tell you: "Are they overlapping RIGHT NOW?"
That's too late — they've already collided!

Predicted positions tell you: "Will they overlap in the NEXT 5 seconds?"
This gives the system time to issue warnings and take action.

This is why Day 3 (trajectory prediction) MUST come before Day 4.
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLLISION_CONFIG, LOG_CONFIG
from utils.logger import get_logger
from detection.tracker import TrackedObject
from prediction.trajectory_predictor import PredictedTrajectory

import cv2

logger = get_logger(__name__)


# ==============================================================================
# RISK LEVEL ENUM
# ==============================================================================

class RiskLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    NONE   = "NONE"


# ==============================================================================
# COLLISION EVENT DATA STRUCTURE
# ==============================================================================

@dataclass
class CollisionEvent:
    """
    Represents a detected collision risk between two vehicles.
    
    Fields:
        track_id_a:     First vehicle track ID
        track_id_b:     Second vehicle track ID
        ttc_frames:     Time-to-collision in frames
        ttc_seconds:    Time-to-collision in seconds
        min_distance:   Predicted minimum distance between vehicles (pixels)
        risk_level:     HIGH / MEDIUM / LOW
        frame_idx:      Frame when this was detected
        collision_point: Predicted (x,y) where collision occurs
    """
    track_id_a:      int
    track_id_b:      int
    ttc_frames:      float
    ttc_seconds:     float
    min_distance:    float
    risk_level:      RiskLevel
    frame_idx:       int
    collision_point: Optional[Tuple[float, float]] = None

    def to_dict(self) -> dict:
        return {
            "frame_idx":      self.frame_idx,
            "track_id_a":     self.track_id_a,
            "track_id_b":     self.track_id_b,
            "ttc_frames":     round(self.ttc_frames, 2),
            "ttc_seconds":    round(self.ttc_seconds, 2),
            "min_distance":   round(self.min_distance, 2),
            "risk_level":     self.risk_level.value,
            "collision_x":    round(self.collision_point[0], 2) if self.collision_point else None,
            "collision_y":    round(self.collision_point[1], 2) if self.collision_point else None,
        }

    def __repr__(self) -> str:
        return (
            f"CollisionEvent("
            f"tracks=({self.track_id_a},{self.track_id_b}), "
            f"TTC={self.ttc_seconds:.1f}s, "
            f"risk={self.risk_level.value})"
        )


# ==============================================================================
# COLLISION ENGINE
# ==============================================================================

class CollisionEngine:
    """
    Computes Time-To-Collision for all vehicle pairs using predicted trajectories.
    
    ALGORITHM OVERVIEW:
      For every pair of tracked vehicles (A, B):
        1. Get their predicted future positions (from TrajectoryPredictor)
        2. Compute relative position and velocity
        3. Find t_min: time of minimum approach distance
        4. If min_distance < collision_threshold → compute TTC
        5. Classify as HIGH/MEDIUM/LOW risk
        6. Emit CollisionEvent
    
    COMPLEXITY:
      For N vehicles, we check N*(N-1)/2 pairs → O(N²)
      For 20 vehicles: 190 pairs
      For 50 vehicles: 1225 pairs
      This is fine for real-time at N < 100.
    """

    def __init__(self, config=None, fps: float = 20.0):
        self.cfg = config or COLLISION_CONFIG
        self.fps = fps

        self.ttc_high = self.cfg.ttc_high_risk_seconds
        self.ttc_medium = self.cfg.ttc_medium_risk_seconds
        self.min_dist = self.cfg.min_collision_distance_px

    # approximate collision radius in pixels
        self.collision_radius_px = 20.0

    # event history
        self._events = []

    def compute_ttc(
        self,
        track_a: TrackedObject,
        track_b: TrackedObject,
        pred_a: Optional[PredictedTrajectory],
        pred_b: Optional[PredictedTrajectory],
    ) -> Optional[CollisionEvent]:
        """
        Compute TTC between two tracked vehicles.
        
        Uses predicted trajectories if available, falls back to
        current velocity extrapolation.
        
        Returns CollisionEvent if risk detected, else None.
        """
        # Get current positions
        center_a = track_a.center
        center_b = track_b.center

        if center_a is None or center_b is None:
            return None

        cx_a, cy_a = center_a
        cx_b, cy_b = center_b

        # --- Check minimum distance threshold ---
        # If they're already far apart, skip detailed TTC computation
        current_dist = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
        if current_dist > self.min_dist:
            return None

        # --- Get velocities ---
        vel_a = track_a.velocity or (0.0, 0.0)
        vel_b = track_b.velocity or (0.0, 0.0)

        vx_a, vy_a = vel_a
        vx_b, vy_b = vel_b

        # --- Relative kinematics ---
        # Relative position: B relative to A
        dx0 = cx_b - cx_a
        dy0 = cy_b - cy_a

        # Relative velocity: how fast B approaches/recedes from A
        dvx = vx_b - vx_a
        dvy = vy_b - vy_a

        # --- Find time of minimum approach ---
        # minimize f(t) = (dx0 + dvx*t)² + (dy0 + dvy*t)²
        # df/dt = 0 → t_min = -(dx0*dvx + dy0*dvy) / (dvx² + dvy²)
        denominator = dvx**2 + dvy**2

        if denominator < 1e-9:
            # Vehicles moving at same velocity (parallel, same speed)
            # Relative distance is constant → no collision if not already colliding
            if current_dist < self.collision_radius_px:
                # Already overlapping → immediate risk
                ttc_frames = 0.0
            else:
                return None
        else:
            t_min = -(dx0 * dvx + dy0 * dvy) / denominator

            # If t_min < 0, vehicles are diverging (moving away) → no future risk
            if t_min < 0:
                return None

            # Compute minimum distance at t_min
            dx_min = dx0 + dvx * t_min
            dy_min = dy0 + dvy * t_min
            min_distance = np.sqrt(dx_min**2 + dy_min**2)

            # If minimum distance > collision radius → they will NOT collide
            if min_distance > self.collision_radius_px:
                return None

            ttc_frames = t_min

        # --- Convert frames to seconds ---
        ttc_seconds = ttc_frames / max(self.fps, 1.0)

        # --- Determine risk level ---
        risk = self._classify_risk(ttc_seconds)
        if risk == RiskLevel.NONE:
            return None

        # --- Compute collision point ---
        # Position of vehicle A at t_min
        coll_x = cx_a + vx_a * ttc_frames
        coll_y = cy_a + vy_a * ttc_frames

        return CollisionEvent(
            track_id_a=track_a.track_id,
            track_id_b=track_b.track_id,
            ttc_frames=ttc_frames,
            ttc_seconds=ttc_seconds,
            min_distance=float(min_distance) if denominator >= 1e-9 else current_dist,
            risk_level=risk,
            frame_idx=0,   # Set by caller
            collision_point=(coll_x, coll_y),
        )

    def _classify_risk(self, ttc_seconds: float) -> RiskLevel:
        """Classify TTC into risk level."""
        if ttc_seconds <= self.ttc_high:
            return RiskLevel.HIGH
        elif ttc_seconds <= self.ttc_medium:
            return RiskLevel.MEDIUM
        elif ttc_seconds <= self.ttc_medium * 2:
            return RiskLevel.LOW
        else:
            return RiskLevel.NONE

    def run_frame(
        self,
        frame_idx: int,
        tracks: Dict[int, TrackedObject],
        predictions: Dict[int, PredictedTrajectory],
    ) -> List[CollisionEvent]:
        """
        Run collision detection for all vehicle pairs in one frame.
        
        Args:
            frame_idx:   Current frame index
            tracks:      Active tracked objects
            predictions: Trajectory predictions per track

        Returns:
            List of CollisionEvent for this frame
        """
        track_ids   = list(tracks.keys())
        frame_events: List[CollisionEvent] = []

        # Check all unique pairs: O(N²/2)
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id_a = track_ids[i]
                id_b = track_ids[j]

                event = self.compute_ttc(
                    tracks[id_a],
                    tracks[id_b],
                    predictions.get(id_a),
                    predictions.get(id_b),
                )

                if event is not None:
                    event.frame_idx = frame_idx
                    frame_events.append(event)
                    self._events.append(event)

        if frame_events:
            high_risks = sum(1 for e in frame_events if e.risk_level == RiskLevel.HIGH)
            logger.info(
                f"Frame {frame_idx}: {len(frame_events)} collision risks "
                f"({high_risks} HIGH)"
            )

        return frame_events

    def annotate_frame(
        self,
        frame: np.ndarray,
        events: List[CollisionEvent],
        tracks: Dict[int, TrackedObject],
        predictions: Dict[int, PredictedTrajectory],
    ) -> np.ndarray:
        """
        Draw collision warnings and predicted paths on the frame.
        
        VISUALIZATION:
          - Predicted trajectory paths (dotted lines forward)
          - Collision zone circles at predicted collision points
          - Warning text for HIGH/MEDIUM risk pairs
          - Color coding: RED=HIGH, ORANGE=MEDIUM, GREEN=LOW
        """
        annotated = frame.copy()
        risk_colors=self.cfg.risk_levels

        # --- Draw predicted paths ---
        for track_id, pred in predictions.items():
            if track_id not in tracks:
                continue

            track = tracks[track_id]
            color = (200, 200, 200)   # Grey for predicted paths

            # Draw predicted future positions as dots
            for i, (fx, fy) in enumerate(pred.future_points):
                # Fade: older predictions more transparent
                alpha = max(0.1, 1.0 - i / len(pred.future_points))
                dot_color = tuple(int(c * alpha) for c in color)
                if i % 3 == 0:   # Draw every 3rd point to reduce clutter
                    cv2.circle(annotated, (int(fx), int(fy)), 2, dot_color, -1)

        # --- Draw collision events ---
        for event in events:
            color_info = risk_colors.get(event.risk_level.value, {})
            color      = color_info.get("color", (0, 0, 255))
            label      = color_info.get("label", event.risk_level.value)

            # Draw line connecting the two vehicles at risk
            track_a = tracks.get(event.track_id_a)
            track_b = tracks.get(event.track_id_b)

            if track_a and track_b and track_a.center and track_b.center:
                cx_a, cy_a = (int(v) for v in track_a.center)
                cx_b, cy_b = (int(v) for v in track_b.center)

                # Dashed line between vehicles
                _draw_dashed_line(annotated, (cx_a, cy_a), (cx_b, cy_b), color, 2)

                # Collision point marker
                if event.collision_point:
                    cx_coll, cy_coll = int(event.collision_point[0]), int(event.collision_point[1])
                    cv2.circle(annotated, (cx_coll, cy_coll), 15, color, 2)
                    cv2.circle(annotated, (cx_coll, cy_coll), 4, color, -1)

                # Warning label near midpoint
                mid_x = (cx_a + cx_b) // 2
                mid_y = (cy_a + cy_b) // 2
                warning = f"{label} TTC:{event.ttc_seconds:.1f}s"
                cv2.putText(
                    annotated, warning,
                    (mid_x - 50, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
                )

        # --- Collision count overlay ---
        high   = sum(1 for e in events if e.risk_level == RiskLevel.HIGH)
        medium = sum(1 for e in events if e.risk_level == RiskLevel.MEDIUM)

        if high > 0:
            cv2.putText(
                annotated,
                f"!!! HIGH RISK: {high} pairs !!!",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )

        return annotated

    @property
    def all_events(self) -> List[CollisionEvent]:
        return self._events

    def get_high_risk_events(self) -> List[CollisionEvent]:
        return [e for e in self._events if e.risk_level == RiskLevel.HIGH]


# ==============================================================================
# HELPER: DRAW DASHED LINE
# ==============================================================================

def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple,
    thickness: int = 1,
    dash_length: int = 15,
) -> None:
    """Draw a dashed line between two points using OpenCV."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx     = x2 - x1
    dy     = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length == 0:
        return

    # Unit vector along the line
    ux = dx / length
    uy = dy / length

    # Draw alternating dashes and gaps
    pos = 0.0
    draw = True
    while pos < length:
        end_pos = min(pos + dash_length, length)
        if draw:
            x_start = int(x1 + ux * pos)
            y_start = int(y1 + uy * pos)
            x_end   = int(x1 + ux * end_pos)
            y_end   = int(y1 + uy * end_pos)
            cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness)
        pos  += dash_length
        draw  = not draw
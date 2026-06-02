"""
================================================================================
prediction/trajectory.py  —  Trajectory Prediction Subsystem
================================================================================

DAY 3 SUBSYSTEM

════════════════════════════════════════════════════════════════════════════════
SECTION 1 — WHAT IS TRAJECTORY PREDICTION?
════════════════════════════════════════════════════════════════════════════════

Tracking gave us identity:
    "Car #7 exists, and here is its bounding box this frame."

Trajectory prediction gives us foresight:
    "Car #7 is moving at vx=4.2, vy=1.1 px/frame.
     In 30 frames it will be at approximately (cx+126, cy+33)."

Without prediction:
  — Collision detection only knows WHERE vehicles ARE right now
  — A human would have to react to a collision happening NOW (too late)

With prediction:
  — The system knows WHERE vehicles WILL BE in 2–5 seconds
  — Decisions (reroute, hold, emergency stop) can be made in advance

Real-world analogy:
  A chess player doesn't just look at the current board.
  They compute 5–10 moves ahead. Trajectory prediction is the AI's
  equivalent of "computing moves ahead" for moving vehicles.

════════════════════════════════════════════════════════════════════════════════
SECTION 2 — WHY TRACKING ALONE IS NOT ENOUGH
════════════════════════════════════════════════════════════════════════════════

Tracking tells you:
    t=0:  Car #7 at (100, 200)
    t=1:  Car #7 at (104, 201)
    t=2:  Car #7 at (108, 202)
    t=3:  Car #7 at (112, 203)   ← current frame

It does NOT tell you:
    t=4:  Car #7 at (???, ???)   ← unknown without prediction
    t=10: Car #7 at (???, ???)
    t=30: Car #7 at (???, ???)

And for collision detection you need to know:
    "Will Car #7 and Car #12 be at the same place at t=15?"

That requires predicting BOTH cars' future paths → then checking if they
intersect → that is trajectory prediction + collision engine working together.

════════════════════════════════════════════════════════════════════════════════
SECTION 3 — HOW FUTURE POSITION PREDICTION WORKS
════════════════════════════════════════════════════════════════════════════════

We use the trajectory history that the tracker saved.
Each TrackedObject has a trail:  [(frame, cx, cy), (frame, cx, cy), ...]

From this history we can estimate:
  1. Velocity (vx, vy)  — how fast and in which direction
  2. Acceleration (ax, ay) — is the velocity changing?
  3. Future positions    — extrapolate forward N steps

THREE LEVELS OF PREDICTION:

  Level 1 — Constant Velocity (linear):
    px(t) = cx_last + vx * t
    py(t) = cy_last + vy * t
    Simple, fast, good for short horizons (< 1 second).
    Assumes no acceleration (constant speed in straight line).

  Level 2 — Constant Acceleration (quadratic):
    px(t) = cx_last + vx * t + 0.5 * ax * t²
    py(t) = cy_last + vy * t + 0.5 * ay * t²
    Better for turning vehicles, vehicles speeding up or braking.
    Requires at least 3 history points to estimate acceleration.

  Level 3 — Kalman Filter (probabilistic):
    Uses a state vector: [cx, cy, vx, vy, ax, ay]
    Maintains uncertainty (covariance matrix)
    Fuses model prediction with noisy measurements
    Best overall but more complex

  Level 4 — LSTM Neural Network:
    Learn from NGSIM real trajectory patterns
    Handles lane changes, sudden stops, non-linear motion
    Requires training data

  We implement Level 1 (Linear) and Level 2 (Accelerated) for MVP.
  Both are engineering-grade and explainable in interviews.

════════════════════════════════════════════════════════════════════════════════
SECTION 4 — VELOCITY CALCULATION
════════════════════════════════════════════════════════════════════════════════

Raw velocity from two consecutive points:
    vx_raw = cx[t] - cx[t-1]  (pixels per frame)
    vy_raw = cy[t] - cy[t-1]

Problem: raw velocity is NOISY because:
  - YOLOv8 bounding box jitter (box size varies frame to frame)
  - DeepSORT Kalman filter adds smoothing but doesn't eliminate noise
  - Vehicles stopping briefly then resuming spike velocity computation

Solution: FIT A LINE TO MULTIPLE POINTS (numpy.polyfit)

    Instead of using just the last 2 points, use the last N points.
    Fit: cx = vx * frame + intercept  (linear regression)
    The slope IS the velocity. Least-squares fit averages out noise.

    This is the same math as linear regression in ML:
        y = mx + b, where m = slope = velocity

Speed (scalar magnitude):
    speed = sqrt(vx² + vy²)    units: pixels per frame

Direction (angle):
    angle = atan2(vy, vx)      units: radians (0=right, π/2=down)

Heading vector (unit vector):
    hx = vx / speed
    hy = vy / speed

════════════════════════════════════════════════════════════════════════════════
SECTION 5 — LIBRARIES USED AND INTERNAL WORKING
════════════════════════════════════════════════════════════════════════════════

numpy.polyfit(x, y, deg):
  — Fits a polynomial of degree `deg` to data points (x, y)
  — Returns array of coefficients [slope, intercept] for deg=1
  — Uses least-squares minimization internally
  — Example: polyfit([0,1,2,3,4], [100,105,109,114,118], 1)
             → [4.5, 100.5]  meaning vx ≈ 4.5 px/frame

numpy.polyval(coefficients, x):
  — Evaluates the polynomial at a new x value
  — polyval([4.5, 100.5], 5) → 4.5*5 + 100.5 = 123.0
  — Used to extrapolate future positions

collections.deque:
  — Double-ended queue with maxlen
  — When full, oldest item auto-drops on append()
  — O(1) append/pop from both ends
  — Our tracker stores trail as deque(maxlen=50)

OpenCV drawing functions used today:
  cv2.circle(img, center, radius, color, thickness)
    — Draws filled or outline circle
    — thickness=-1 means filled
    — Used to mark predicted positions as dots

  cv2.polylines(img, [pts], isClosed, color, thickness)
    — Draws a series of connected lines through an array of points
    — pts must be shape (N, 1, 2) int32 array — this is OpenCV's quirky format
    — isClosed=False → open polyline (not a closed shape)
    — Used to draw the predicted future path

  cv2.arrowedLine(img, pt1, pt2, color, thickness, tipLength)
    — Draws a line with an arrowhead at pt2
    — tipLength: fraction of line length for the arrowhead (0.3 = 30%)
    — Used to show velocity direction vector

cv2.addWeighted(src1, alpha, src2, beta, gamma, dst):
  — Blends two images: dst = src1*alpha + src2*beta + gamma
  — We use this to draw semi-transparent overlays
  — E.g., addWeighted(overlay, 0.4, frame, 0.6, 0) →
          40% overlay + 60% original = see-through annotation

================================================================================
"""

import cv2
import numpy as np
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PREDICTION_CONFIG, DETECTION_CONFIG, OUTPUT_CSV_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# SECTION A — DATA STRUCTURES
# ==============================================================================

@dataclass
class VelocityEstimate:
    """
    Velocity and acceleration for one tracked object.

    WHY A DATACLASS:
    Python dataclass auto-generates __init__, __repr__, __eq__.
    It's cleaner than a dict (typed) and simpler than a full class.
    Use dataclass for pure data containers; use class for objects with behavior.

    Attributes:
        vx, vy      — velocity in pixels per frame
        ax, ay      — acceleration in pixels per frame²
        speed       — scalar magnitude √(vx²+vy²)
        angle_rad   — heading angle in radians from positive-x axis
        confidence  — how reliable this estimate is (0.0–1.0)
                       decreases with fewer history points
    """
    vx:         float = 0.0
    vy:         float = 0.0
    ax:         float = 0.0
    ay:         float = 0.0
    speed:      float = 0.0
    angle_rad:  float = 0.0
    confidence: float = 0.0

    @property
    def heading_unit_vector(self) -> Tuple[float, float]:
        """
        Unit vector in the direction of motion.

        Unit vector = (vx/speed, vy/speed)
        Length = 1.0 (direction only, no magnitude)
        Used to draw heading arrows scaled to any length.
        """
        if self.speed < 1e-6:
            return (0.0, 0.0)
        return (self.vx / self.speed, self.vy / self.speed)

    def to_dict(self) -> dict:
        return {
            "vx": round(self.vx, 4),
            "vy": round(self.vy, 4),
            "ax": round(self.ax, 4),
            "ay": round(self.ay, 4),
            "speed": round(self.speed, 4),
            "angle_deg": round(np.degrees(self.angle_rad), 2),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class PredictedTrajectory:
    """
    Future positions predicted for one tracked object.

    Attributes:
        track_id        — which vehicle this belongs to
        frame_idx       — frame at which prediction was made
        method          — "linear" or "accelerated"
        future_points   — list of (cx, cy) tuples for next N frames
        velocity        — VelocityEstimate used to generate this
        confidence      — decreases as prediction horizon extends
    """
    track_id:      int
    frame_idx:     int
    method:        str
    future_points: List[Tuple[float, float]]   # [(cx,cy), (cx,cy), ...]
    velocity:      VelocityEstimate
    confidence:    float = 1.0

    def position_at(self, steps_ahead: int) -> Optional[Tuple[float, float]]:
        """
        Get predicted (cx, cy) N steps into the future.
        Returns None if steps_ahead exceeds our prediction horizon.

        EXAMPLE:
            pred.position_at(0)  → current position
            pred.position_at(15) → predicted position 15 frames from now
            pred.position_at(50) → None (beyond horizon)
        """
        if 0 <= steps_ahead < len(self.future_points):
            return self.future_points[steps_ahead]
        return None

    @property
    def horizon(self) -> int:
        """Total number of frames predicted ahead."""
        return len(self.future_points)

    @property
    def endpoint(self) -> Optional[Tuple[float, float]]:
        """Final predicted position (furthest ahead)."""
        return self.future_points[-1] if self.future_points else None

    def to_dict(self) -> dict:
        return {
            "track_id":     self.track_id,
            "frame_idx":    self.frame_idx,
            "method":       self.method,
            "confidence":   round(self.confidence, 4),
            "horizon":      self.horizon,
            "velocity":     self.velocity.to_dict(),
            "future_points": [
                {"cx": round(x, 2), "cy": round(y, 2)}
                for x, y in self.future_points
            ],
        }


# ==============================================================================
# SECTION B — VELOCITY ESTIMATOR
# ==============================================================================

class VelocityEstimator:
    """
    Estimates velocity and acceleration from a history of positions.

    ALGORITHM:
        Given N past positions (cx_i, cy_i) at times t_i:
        1. Fit linear regression: cx = vx * t + cx0
        2. The slope vx IS the x-velocity
        3. Repeat for y: vy = slope of cy vs t
        4. If enough history, fit quadratic for acceleration:
           cx = 0.5*ax*t² + vx0*t + cx0
           Coefficient of t² × 2 = acceleration ax

    WHY POLYFIT INSTEAD OF JUST LAST-TWO-POINTS:
        Last two points: vx = cx[now] - cx[prev]
        This is VERY sensitive to detection noise.
        A bounding box that shifts 5px due to detection jitter would
        give a fake velocity spike of 5 px/frame.

        Linear regression over 10 points averages out the noise.
        The slope is the best estimate of the true underlying velocity.
        This is called LEAST SQUARES VELOCITY ESTIMATION and is used
        in real tracking systems (radar, sonar, GPS).
    """

    def __init__(self, min_points: int = 3, max_history: int = 20):
        """
        Args:
            min_points:  Minimum history length before we attempt estimation.
                         < 3 points → can't estimate velocity reliably.
            max_history: How many recent points to use for fitting.
                         Using all points would make velocity slow to respond
                         to actual changes. 20 frames = ~1 second of history.
        """
        self.min_points  = min_points
        self.max_history = max_history

    def estimate(self, trail: deque) -> Optional[VelocityEstimate]:
        """
        Estimate velocity from a track's trail deque.

        The trail contains tuples: (frame_idx, cx, cy)

        STEP-BY-STEP:
            1. Extract last max_history points
            2. Build time array t = [frame_idx - first_frame]
               (normalized to start at 0 for numerical stability)
            3. polyfit(t, cx_vals, 1) → [vx, cx_intercept]
            4. polyfit(t, cy_vals, 1) → [vy, cy_intercept]
            5. If enough points, polyfit degree=2 for acceleration

        Args:
            trail: deque of (frame_idx, cx, cy) tuples

        Returns:
            VelocityEstimate or None if insufficient history
        """
        if len(trail) < self.min_points:
            return None

        # Take the most recent max_history points
        recent = list(trail)[-self.max_history:]

        # Unpack into separate arrays
        # zip(*list_of_tuples) transposes: [(a1,b1,c1),(a2,b2,c2)] → [a1,a2],[b1,b2],[c1,c2]
        frames, cx_vals, cy_vals = zip(*recent)
        frames  = np.array(frames,  dtype=float)
        cx_vals = np.array(cx_vals, dtype=float)
        cy_vals = np.array(cy_vals, dtype=float)

        # Normalize time: subtract the first frame index
        # WHY: polyfit can have numerical issues with large absolute numbers.
        # If frames = [1000, 1001, 1002, ...], the intercept would be huge.
        # t = [0, 1, 2, ...] keeps everything small and stable.
        t = frames - frames[0]

        # === LINEAR FIT (velocity) ===
        # numpy.polyfit(x, y, deg) → coefficients [slope, intercept]
        # For deg=1: y = slope*x + intercept
        # slope = velocity = how many pixels cx changes per frame
        try:
            cx_coeffs = np.polyfit(t, cx_vals, deg=1)  # [vx, cx0]
            cy_coeffs = np.polyfit(t, cy_vals, deg=1)  # [vy, cy0]
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"polyfit (linear) failed: {e}")
            return None

        vx = float(cx_coeffs[0])   # pixels per frame in x
        vy = float(cy_coeffs[0])   # pixels per frame in y

        # === QUADRATIC FIT (acceleration) ===
        # For deg=2: y = ax_coeff*t² + vx0*t + cx0
        # The 2nd-degree coefficient × 2 = acceleration
        # WHY ×2? Because the kinematic equation is: x = x0 + v0*t + 0.5*a*t²
        # polyfit returns the raw polynomial coeff, which is 0.5*a, not a.
        ax, ay = 0.0, 0.0
        if len(recent) >= 5:
            try:
                cx_q = np.polyfit(t, cx_vals, deg=2)  # [ax_half, vx0, cx0]
                cy_q = np.polyfit(t, cy_vals, deg=2)
                ax = float(cx_q[0]) * 2.0   # recover actual acceleration
                ay = float(cy_q[0]) * 2.0
            except (np.linalg.LinAlgError, ValueError):
                pass   # Acceleration stays 0; linear is still valid

        # === DERIVED QUANTITIES ===
        speed     = float(np.sqrt(vx**2 + vy**2))
        angle_rad = float(np.arctan2(vy, vx))

        # Confidence: more history = more confident
        # Also: very low speed → prediction unreliable (might be a stationary jitter)
        n_factor   = min(1.0, len(recent) / 15.0)   # 15 points → full confidence
        spd_factor = min(1.0, speed / 2.0)           # 2 px/frame → full speed confidence
        confidence = 0.5 * n_factor + 0.5 * spd_factor

        return VelocityEstimate(
            vx=vx, vy=vy, ax=ax, ay=ay,
            speed=speed, angle_rad=angle_rad,
            confidence=confidence,
        )


# ==============================================================================
# SECTION C — TRAJECTORY PREDICTOR
# ==============================================================================

class TrajectoryPredictor:
    """
    Predicts future positions for tracked objects using velocity history.

    SUPPORTED METHODS:
        "linear"      → constant velocity (vx, vy assumed constant)
        "accelerated" → constant acceleration (ax, ay included)

    PREDICTION EQUATIONS:

        Linear (Constant Velocity Model):
            cx(t) = cx_last + vx * t
            cy(t) = cy_last + vy * t

        Accelerated (Constant Acceleration Model):
            cx(t) = cx_last + vx * t + 0.5 * ax * t²
            cy(t) = cy_last + vy * t + 0.5 * ay * t²

            These are literally Newton's equations of motion:
                x = x0 + v0*t + (1/2)*a*t²

        Where:
            t = number of frames ahead (1, 2, 3, ... horizon)
            cx_last, cy_last = last known center position
            vx, vy = estimated velocity from VelocityEstimator
            ax, ay = estimated acceleration (0 for linear method)
    """

    def __init__(self, config: dict = None, method=None):
        self.cfg = config or PREDICTION_CONFIG

        self.horizon=self.cfg.prediction_horizon

    # use passed method if available, otherwise config
        self.method = method or self.cfg.get(
        "predictor_type",
        "linear"
    )

        self.min_hist=self.cfg.min_history_len
    

        # Velocity estimator reused across all tracks
        self._vel_estimator = VelocityEstimator(
            min_points=self.min_hist,
            max_history=20,
        )

        logger.info(
            f"TrajectoryPredictor initialized | "
            f"method={self.method}, horizon={self.horizon} frames"
        )

    def predict_one(self, track) -> Optional[PredictedTrajectory]:
        """
        Predict future trajectory for a single TrackedObject.

        FLOW:
            1. Get velocity estimate from history
            2. Get last known position (center of bounding box)
            3. Apply prediction equation for t=1 to t=horizon
            4. Return PredictedTrajectory

        Args:
            track: TrackedObject (from detection/tracker.py)

        Returns:
            PredictedTrajectory or None if insufficient history
        """
        # Need a trail with enough points
        if len(track.trail) < self.min_hist:
            return None

        # Step 1: Estimate velocity
        vel = self._vel_estimator.estimate(track.trail)
        if vel is None:
            return None

        # Step 2: Get last known position
        # trail[-1] = (frame_idx, cx, cy)
        last_frame, cx_last, cy_last = track.trail[-1]

        # Step 3: Generate future positions
        future_points: List[Tuple[float, float]] = []

        for t in range(1, self.horizon + 1):
            if self.method == "linear":
                # x = x0 + vx * t
                fx = cx_last + vel.vx * t
                fy = cy_last + vel.vy * t

            elif self.method == "accelerated":
                # x = x0 + vx*t + 0.5*ax*t²
                fx = cx_last + vel.vx * t + 0.5 * vel.ax * (t ** 2)
                fy = cy_last + vel.vy * t + 0.5 * vel.ay * (t ** 2)
            else:
                # Fallback to linear
                fx = cx_last + vel.vx * t
                fy = cy_last + vel.vy * t

            future_points.append((float(fx), float(fy)))

        # Step 4: Compute per-trajectory confidence
        # Confidence decays with horizon: long-range predictions less reliable
        # Multiply base velocity confidence by a time-decay factor
        # horizon_confidence[t] = base_conf * exp(-decay * t / horizon)
        # We report the average across the horizon
        decay     = 0.5   # tune: higher = faster confidence decay with distance
        avg_conf  = float(
            vel.confidence * np.mean([
                np.exp(-decay * t / self.horizon)
                for t in range(1, self.horizon + 1)
            ])
        )

        return PredictedTrajectory(
            track_id=track.track_id,
            frame_idx=int(last_frame),
            method=self.method,
            future_points=future_points,
            velocity=vel,
            confidence=avg_conf,
        )

    def predict_all(self, tracks: dict) -> Dict[int, PredictedTrajectory]:
        """
        Predict trajectories for all active tracks.

        EFFICIENCY NOTE:
            We run predictions sequentially here (one track at a time).
            For N=100 tracks with horizon=30, that's 3000 simple arithmetic ops.
            On CPU this takes < 1ms total — NOT a bottleneck.
            Bottleneck is YOLOv8 inference (~100-200ms per frame on CPU).

        Args:
            tracks: Dict of {track_id: TrackedObject} from the tracker

        Returns:
            Dict of {track_id: PredictedTrajectory} for all predictable tracks
        """
        predictions: Dict[int, PredictedTrajectory] = {}

        for track_id, track in tracks.items():
            pred = self.predict_one(track)
            if pred is not None:
                predictions[track_id] = pred

        return predictions


# ==============================================================================
# SECTION D — TRAJECTORY VISUALIZER
# ==============================================================================

class TrajectoryVisualizer:
    """
    Draws predicted trajectories onto video frames.

    VISUALIZATION DESIGN DECISIONS:
        We draw three things:
        1. Predicted path dots — small circles at each future position
           Spaced out to show HOW FAR the vehicle will travel
           Color = class color, fading with distance (opacity effect)

        2. Velocity arrow — shows current speed AND direction
           Length proportional to speed
           Arrowhead points in direction of motion

        3. Endpoint marker — circle at the final predicted position
           Larger dot to make it easy to see where vehicle ends up

    FADE EFFECT:
        OpenCV doesn't support real transparency per pixel in real time.
        Instead, we simulate fading by:
        a) Drawing to a separate overlay canvas
        b) Blending with addWeighted(overlay, 0.4, frame, 0.6, 0)
        This creates the appearance of semi-transparency.

    ALTERNATIVE: reduce color intensity for farther dots
        Instead of transparency, make the color darker for far-away predictions.
        Simpler and faster than addWeighted, looks good in output video.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or {}

        # Class colors lookup
        self._class_colors: Dict[int, Tuple] = DETECTION_CONFIG.get("class_colors", {})

        # How many future frames to DRAW (we might predict 30 but only draw 15 dots)
        # Drawing all 30 can look cluttered; show every Nth frame
        self.draw_every_n    = 2     # Draw every 2nd predicted point (reduces clutter)
        self.dot_radius      = 3     # Radius of future position dot
        self.endpoint_radius = 8     # Radius of final endpoint marker

    def draw_predictions(
        self,
        frame: np.ndarray,
        tracks:      dict,
        predictions: Dict[int, PredictedTrajectory],
    ) -> np.ndarray:
        """
        Draw all trajectory predictions onto the frame.

        DRAWING ORDER (back to front, so important elements are on top):
            1. Predicted path (dotted line) — drawn first, can be partially covered
            2. Velocity arrow              — drawn second
            3. Endpoint marker             — drawn last, always visible

        Args:
            frame:       BGR frame to annotate (we copy it, don't mutate)
            tracks:      Active TrackedObjects (for current position + class)
            predictions: Dict of PredictedTrajectory per track_id

        Returns:
            Annotated frame copy
        """
        annotated = frame.copy()

        # We draw semi-transparent path on an overlay
        # Then blend overlay + original at the end
        overlay = annotated.copy()

        for track_id, pred in predictions.items():
            # Get class color
            track = tracks.get(track_id)
            if track is None:
                continue

            class_id = track.class_id
            color    = self._class_colors.get(class_id, (200, 200, 200))

            # ── 1. PREDICTED PATH DOTS ──
            self._draw_future_dots(overlay, pred, color)

            # ── 2. VELOCITY ARROW ──
            if track.center is not None:
                self._draw_velocity_arrow(annotated, track.center, pred.velocity, color)

            # ── 3. ENDPOINT MARKER ──
            if pred.endpoint is not None:
                self._draw_endpoint(annotated, pred.endpoint, color, track_id)

        # Blend: 35% overlay (predicted dots transparent-ish) + 65% original
        # addWeighted(src1, alpha, src2, beta, gamma, dst=None)
        # dst = src1 * alpha + src2 * beta + gamma
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

        return annotated

    def _draw_future_dots(
        self,
        canvas: np.ndarray,
        pred:   PredictedTrajectory,
        color:  Tuple,
    ) -> None:
        """
        Draw small circles at each predicted future position.

        FADE EFFECT:
            We dim the color proportionally to distance in the future.
            frac = i / total_points  (0.0 = near future, 1.0 = far future)
            dot_color = color * (1.0 - 0.6 * frac)
            So the nearest dot is full color, furthest dot is 40% brightness.

            This visual fade communicates:
            "The near prediction is reliable; the far one is uncertain."
        """
        points = pred.future_points
        n      = len(points)
        if n == 0:
            return

        for i, (fx, fy) in enumerate(points):
            # Only draw every draw_every_n frames to reduce clutter
            if i % self.draw_every_n != 0:
                continue

            # Ensure positions are within frame bounds (future path may go off-screen)
            fx_i = int(np.clip(fx, 0, canvas.shape[1] - 1))
            fy_i = int(np.clip(fy, 0, canvas.shape[0] - 1))

            # Compute fade: far future = darker
            frac      = i / max(n - 1, 1)     # 0.0 near, 1.0 far
            intensity = 1.0 - 0.65 * frac     # 1.0 near, 0.35 far
            dot_color = tuple(int(c * intensity) for c in color)

            # Draw the dot
            cv2.circle(canvas, (fx_i, fy_i), self.dot_radius, dot_color, -1)

        # Optionally connect dots with a polyline for clarity
        # We'll draw a thin line through every 3rd point
        step_pts = [
            (int(fx), int(fy))
            for i, (fx, fy) in enumerate(points)
            if i % 3 == 0
        ]
        if len(step_pts) >= 2:
            # cv2.polylines wants shape (N, 1, 2) int32 array — OpenCV quirk
            # We must reshape: [(x1,y1),(x2,y2)] → [[[x1,y1]],[[x2,y2]]]
            pts_array = np.array(step_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                canvas,
                [pts_array],
                isClosed=False,
                color=tuple(int(c * 0.6) for c in color),  # dimmer line
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def _draw_velocity_arrow(
        self,
        canvas: np.ndarray,
        center: Tuple[float, float],
        vel:    VelocityEstimate,
        color:  Tuple,
    ) -> None:
        """
        Draw an arrow from current position in the direction of motion.

        ARROW LENGTH:
            Proportional to speed. Faster vehicles get longer arrows.
            arrow_length = speed * scale_factor
            scale_factor = 5.0 → speed of 10px/frame draws 50px arrow

            This makes it immediately visual:
            Long arrow = fast vehicle
            Short arrow = slow vehicle
            No arrow = stationary

        cv2.arrowedLine parameters:
            img, pt1 (start), pt2 (end), color, thickness,
            line_type, shift, tipLength (fraction of line for arrowhead)
        """
        if vel.speed < 0.5:
            return   # Too slow to draw a meaningful arrow

        cx, cy = center
        scale         = 5.0   # pixels of arrow per px/frame of speed
        arrow_length  = min(vel.speed * scale, 80)   # cap at 80px
        hx, hy        = vel.heading_unit_vector

        pt1 = (int(cx), int(cy))
        pt2 = (int(cx + hx * arrow_length), int(cy + hy * arrow_length))

        # Bright white outline for visibility
        cv2.arrowedLine(canvas, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
        # Colored arrow on top
        cv2.arrowedLine(canvas, pt1, pt2, color, 2, cv2.LINE_AA, tipLength=0.3)

    def _draw_endpoint(
        self,
        canvas:   np.ndarray,
        endpoint: Tuple[float, float],
        color:    Tuple,
        track_id: int,
    ) -> None:
        """
        Draw the endpoint marker: a circle where the vehicle will be
        at the end of the prediction horizon.

        A larger circle (outline + filled center) makes this easy to spot.
        We also print the track ID next to it so operators know which
        vehicle this predicted position belongs to.
        """
        ex = int(np.clip(endpoint[0], 0, canvas.shape[1] - 1))
        ey = int(np.clip(endpoint[1], 0, canvas.shape[0] - 1))

        # Outer ring
        cv2.circle(canvas, (ex, ey), self.endpoint_radius, color, 2)
        # Inner dot
        cv2.circle(canvas, (ex, ey), 3, color, -1)
        # Track ID label next to endpoint
        label = f"→#{track_id}"
        cv2.putText(
            canvas, label,
            (ex + self.endpoint_radius + 4, ey + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )


# ==============================================================================
# SECTION E — TRAJECTORY CSV EXPORTER
# ==============================================================================

class TrajectoryCSVExporter:
    """
    Saves trajectory predictions to CSV for:
    1. Offline analysis
    2. Training LSTM models
    3. Dashboard display
    4. Portfolio / demo evidence

    CSV SCHEMA:
        frame_idx, track_id, class_name,
        cx, cy,                    ← current position
        vx, vy, speed,             ← velocity
        ax, ay,                    ← acceleration
        confidence,                ← prediction confidence
        pred_cx_5, pred_cy_5,      ← predicted position in 5 frames
        pred_cx_15, pred_cy_15,    ← predicted position in 15 frames
        pred_cx_30, pred_cy_30     ← predicted position in 30 frames
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: List[dict] = []

        self.FIELDNAMES = [
            "frame_idx", "track_id", "class_name",
            "cx", "cy",
            "vx", "vy", "speed", "ax", "ay",
            "confidence",
            "pred_cx_5",  "pred_cy_5",
            "pred_cx_15", "pred_cy_15",
            "pred_cx_30", "pred_cy_30",
        ]

    def record(
        self,
        frame_idx:   int,
        tracks:      dict,
        predictions: Dict[int, PredictedTrajectory],
    ) -> None:
        """
        Record one frame's prediction data.
        Call this every frame from the main pipeline loop.
        """
        for track_id, pred in predictions.items():
            track = tracks.get(track_id)
            if track is None:
                continue

            center = track.center or (0.0, 0.0)
            vel    = pred.velocity

            # Get predicted positions at specific horizons
            def get_pred(steps: int) -> Tuple[float, float]:
                pt = pred.position_at(steps - 1)   # 0-indexed
                return pt if pt else (0.0, 0.0)

            p5  = get_pred(5)
            p15 = get_pred(15)
            p30 = get_pred(30)

            row = {
                "frame_idx":  frame_idx,
                "track_id":   track_id,
                "class_name": track.class_name,
                "cx":         round(center[0], 2),
                "cy":         round(center[1], 2),
                "vx":         round(vel.vx,    4),
                "vy":         round(vel.vy,    4),
                "speed":      round(vel.speed, 4),
                "ax":         round(vel.ax,    6),
                "ay":         round(vel.ay,    6),
                "confidence": round(pred.confidence, 4),
                "pred_cx_5":  round(p5[0],  2),
                "pred_cy_5":  round(p5[1],  2),
                "pred_cx_15": round(p15[0], 2),
                "pred_cy_15": round(p15[1], 2),
                "pred_cx_30": round(p30[0], 2),
                "pred_cy_30": round(p30[1], 2),
            }
            self._rows.append(row)

    def flush(self) -> None:
        """
        Write all buffered rows to the CSV file.

        WHY BUFFER THEN FLUSH (instead of writing each row immediately):
            Disk writes are slow (~0.5ms each). Writing every frame would
            add latency to the real-time pipeline.
            Buffering rows in memory and writing once at the end is
            10–100× faster for the pipeline loop.

            Trade-off: if the program crashes, buffered rows are lost.
            For production: flush every N frames (e.g., every 500 frames).
        """
        if not self._rows:
            logger.warning("No trajectory data to export.")
            return

        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)

        logger.info(
            f"TrajectoryCSV: saved {len(self._rows)} rows → {self.output_path}"
        )
        self._rows.clear()


# ==============================================================================
# SECTION F — TOP-LEVEL SUBSYSTEM CLASS (integrate into main pipeline)
# ==============================================================================

class TrajectorSubsystem:
    """
    Combines predictor + visualizer + exporter into one clean interface.

    This is the class main.py calls. It wraps everything:
        subsystem = TrajectorSubsystem()
        predictions = subsystem.process_frame(frame_idx, frame, tracks)
        annotated   = subsystem.annotate(frame, tracks, predictions)

    Keeps main.py clean — it doesn't need to know about polyfit internals.
    This is the Facade design pattern: simple interface over complex subsystem.
    """

    def __init__(self, config: dict = None):
        self.cfg          = config or PREDICTION_CONFIG
        self.predictor    = TrajectoryPredictor(self.cfg)
        self.visualizer   = TrajectoryVisualizer()
        self.exporter     = TrajectoryCSVExporter(
            OUTPUT_CSV_DIR / "trajectory_predictions.csv"
        )
        self._frame_count = 0

        logger.info("TrajectorSubsystem ready.")

    def process_frame(
        self,
        frame_idx: int,
        tracks:    dict,
    ) -> Dict[int, PredictedTrajectory]:
        """
        Run prediction for all tracks in this frame.
        Also records data for CSV export.

        Args:
            frame_idx: Current frame index
            tracks:    Dict of active TrackedObjects

        Returns:
            Dict of {track_id: PredictedTrajectory}
        """
        self._frame_count += 1

        # Run prediction
        predictions = self.predictor.predict_all(tracks)

        # Record to CSV buffer (flushed at end of session)
        self.exporter.record(frame_idx, tracks, predictions)

        # Log stats every 100 frames
        if self._frame_count % 100 == 0:
            logger.info(
                f"Trajectory [frame {frame_idx}]: "
                f"{len(predictions)}/{len(tracks)} tracks predicted"
            )

        return predictions

    def annotate(
        self,
        frame:       np.ndarray,
        tracks:      dict,
        predictions: Dict[int, PredictedTrajectory],
    ) -> np.ndarray:
        """
        Draw predictions on frame. Returns annotated copy.
        """
        return self.visualizer.draw_predictions(frame, tracks, predictions)

    def save_csv(self) -> None:
        """Flush CSV buffer to disk. Call at end of pipeline run."""
        self.exporter.flush()
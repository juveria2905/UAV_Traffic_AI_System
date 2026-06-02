"""
================================================================================
prediction/motion_models.py — Multi-Model Trajectory Prediction Engine
================================================================================

PURPOSE:
  Implements 5 motion prediction models with automatic model selection
  based on vehicle motion characteristics. Upgrades the existing
  trajectory_predictor.py by providing richer model variants.

MODELS:
  LinearModel:         Constant velocity. Best for smooth straight motion.
  AccelerationModel:   Constant acceleration. Best for vehicles speeding up/braking.
  PolynomialModel:     Polynomial fit. Best for curves and complex motion.
  KalmanModel:         Probabilistic filter. Best for noisy detections.
  LSTMModel:           Neural network. Best for lane changes (requires training).

AUTO-SELECTION LOGIC:
  Compute motion statistics from trail history:
    speed_variance = CV (coefficient of variation) of recent speeds
    mean_accel     = mean absolute acceleration over last N frames

  if speed_variance < 0.20:          → Linear (steady speed)
  elif mean_accel > accel_threshold: → Acceleration (accelerating/braking)
  elif speed_variance < 0.60:        → Polynomial (moderate variation)
  else:                              → Kalman (noisy/erratic)

CONNECTS TO:
  trajectory_predictor.py → TrajectoryPredictor.predict_one() uses these models
  agent_system.py         → model choice logged in reasoning chain
  api.py                 → /api/predictions includes model_type field
================================================================================
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# BASE MODEL INTERFACE
# ==============================================================================

@dataclass
class MotionPrediction:
    """Prediction output from any motion model."""
    model_name:    str
    future_points: List[Tuple[float, float]]   # [(cx,cy), ...]
    confidence:    float
    vx:            float
    vy:            float
    speed:         float
    metadata:      dict = field(default_factory=dict)

    def position_at(self, t: int) -> Optional[Tuple[float, float]]:
        if 0 <= t < len(self.future_points):
            return self.future_points[t]
        return None


class BaseMotionModel(ABC):
    """Abstract base for all motion models."""

    def __init__(self, horizon: int = 30, name: str = "BaseModel"):
        self.horizon = horizon
        self.name    = name

    @abstractmethod
    def predict(
        self,
        trail:     deque,
        horizon:   Optional[int] = None,
    ) -> Optional[MotionPrediction]:
        """
        Predict future positions from a position trail.

        Args:
            trail:   deque of (frame_idx, cx, cy)
            horizon: Number of frames to predict (overrides self.horizon)

        Returns:
            MotionPrediction or None if insufficient history.
        """

    def _extract_arrays(
        self, trail: deque, window: int = 20
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
        """
        Extract normalized time array and position arrays from trail.
        Returns (t, cx, cy, cx_last, cy_last) or None.
        """
        recent = list(trail)[-window:]
        if len(recent) < 3:
            return None

        frames, cx_vals, cy_vals = zip(*recent)
        frames  = np.array(frames,  dtype=float)
        cx_vals = np.array(cx_vals, dtype=float)
        cy_vals = np.array(cy_vals, dtype=float)

        t = frames - frames[0]   # normalize to start at 0
        cx_last = float(cx_vals[-1])
        cy_last = float(cy_vals[-1])

        return t, cx_vals, cy_vals, cx_last, cy_last

    def _confidence_decay(self, base_conf: float, horizon: int) -> float:
        """Exponential confidence decay over prediction horizon."""
        decay_rate = 0.5
        return float(base_conf * np.mean([
            np.exp(-decay_rate * t / max(horizon, 1))
            for t in range(1, horizon + 1)
        ]))


# ==============================================================================
# 1. LINEAR MODEL — Constant Velocity
# ==============================================================================

class LinearModel(BaseMotionModel):
    """
    Constant velocity model.
    Estimates vx, vy via linear regression (polyfit deg=1) over trail.
    Extrapolates: cx(t) = cx_last + vx * t

    BEST FOR: highway-style straight motion, steady speed.
    WORST FOR: curves, stop-and-go, lane changes.
    """

    def __init__(self, horizon: int = 30, max_history: int = 20):
        super().__init__(horizon, "LinearModel")
        self._max_history = max_history

    def predict(self, trail: deque, horizon: Optional[int] = None) -> Optional[MotionPrediction]:
        h = horizon or self.horizon
        arrays = self._extract_arrays(trail, self._max_history)
        if arrays is None:
            return None

        t, cx_vals, cy_vals, cx_last, cy_last = arrays

        try:
            cx_coeffs = np.polyfit(t, cx_vals, deg=1)
            cy_coeffs = np.polyfit(t, cy_vals, deg=1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        vx = float(cx_coeffs[0])
        vy = float(cy_coeffs[0])
        speed = float(np.sqrt(vx**2 + vy**2))

        future = [
            (cx_last + vx * t_step, cy_last + vy * t_step)
            for t_step in range(1, h + 1)
        ]

        n_factor   = min(1.0, len(list(trail)) / 15.0)
        spd_factor = min(1.0, speed / 2.0)
        base_conf  = 0.5 * n_factor + 0.5 * spd_factor

        return MotionPrediction(
            model_name="LinearModel",
            future_points=future,
            confidence=self._confidence_decay(base_conf, h),
            vx=vx, vy=vy, speed=speed,
            metadata={"slope_cx": float(cx_coeffs[0]), "slope_cy": float(cy_coeffs[0])},
        )


# ==============================================================================
# 2. ACCELERATION MODEL — Constant Acceleration
# ==============================================================================

class AccelerationModel(BaseMotionModel):
    """
    Constant acceleration model (Newton's equations of motion).
    cx(t) = cx_last + vx*t + 0.5*ax*t²

    BEST FOR: vehicles decelerating to stop, accelerating from stop.
    WORST FOR: uniform-speed vehicles (over-fits noise as acceleration).
    """

    def __init__(self, horizon: int = 30, max_history: int = 20):
        super().__init__(horizon, "AccelerationModel")
        self._max_history = max_history

    def predict(self, trail: deque, horizon: Optional[int] = None) -> Optional[MotionPrediction]:
        h = horizon or self.horizon
        arrays = self._extract_arrays(trail, self._max_history)
        if arrays is None or len(list(trail)) < 5:
            return None

        t, cx_vals, cy_vals, cx_last, cy_last = arrays

        try:
            cx_q = np.polyfit(t, cx_vals, deg=2)
            cy_q = np.polyfit(t, cy_vals, deg=2)
        except (np.linalg.LinAlgError, ValueError):
            return None

        # polyfit deg=2 returns [a/2, v0, x0] (physics: x = 0.5*a*t² + v0*t + x0)
        ax = float(cx_q[0]) * 2.0
        ay = float(cy_q[0]) * 2.0
        vx = float(cx_q[1])
        vy = float(cy_q[1])
        speed = float(np.sqrt(vx**2 + vy**2))

        future = [
            (
                cx_last + vx * t_step + 0.5 * ax * t_step**2,
                cy_last + vy * t_step + 0.5 * ay * t_step**2,
            )
            for t_step in range(1, h + 1)
        ]

        base_conf = min(1.0, len(list(trail)) / 20.0) * 0.85
        return MotionPrediction(
            model_name="AccelerationModel",
            future_points=future,
            confidence=self._confidence_decay(base_conf, h),
            vx=vx, vy=vy, speed=speed,
            metadata={"ax": round(ax, 4), "ay": round(ay, 4)},
        )


# ==============================================================================
# 3. POLYNOMIAL MODEL — Higher-degree curve fitting
# ==============================================================================

class PolynomialModel(BaseMotionModel):
    """
    Polynomial regression model (degree 3 or 4).
    Captures curves, S-bends, and non-linear motion.

    BEST FOR: vehicles making turns, intersection navigation.
    WORST FOR: straight-line motion (overfits), sparse history.
    """

    def __init__(self, horizon: int = 30, degree: int = 3, max_history: int = 20):
        super().__init__(horizon, "PolynomialModel")
        self._degree      = degree
        self._max_history = max_history

    def predict(self, trail: deque, horizon: Optional[int] = None) -> Optional[MotionPrediction]:
        h = horizon or self.horizon
        trail_list = list(trail)
        if len(trail_list) < max(self._degree + 2, 6):
            return None

        arrays = self._extract_arrays(trail, self._max_history)
        if arrays is None:
            return None

        t, cx_vals, cy_vals, cx_last, cy_last = arrays

        try:
            cx_coeffs = np.polyfit(t, cx_vals, deg=self._degree)
            cy_coeffs = np.polyfit(t, cy_vals, deg=self._degree)
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Evaluate fit at last known time to get velocity (derivative at t_last)
        t_last = t[-1]
        vx = float(np.polyval(np.polyder(cx_coeffs), t_last))
        vy = float(np.polyval(np.polyder(cy_coeffs), t_last))
        speed = float(np.sqrt(vx**2 + vy**2))

        future = []
        for t_step in range(1, h + 1):
            t_eval = t_last + t_step
            fx = float(np.polyval(cx_coeffs, t_eval))
            fy = float(np.polyval(cy_coeffs, t_eval))
            future.append((fx, fy))

        base_conf = min(1.0, len(trail_list) / 20.0) * 0.80
        return MotionPrediction(
            model_name=f"PolynomialModel(deg={self._degree})",
            future_points=future,
            confidence=self._confidence_decay(base_conf, h),
            vx=vx, vy=vy, speed=speed,
            metadata={"degree": self._degree},
        )


# ==============================================================================
# 4. KALMAN MODEL — Probabilistic tracking filter
# ==============================================================================

class KalmanModel(BaseMotionModel):
    """
    Simplified Kalman Filter for trajectory prediction.

    State vector: [cx, cy, vx, vy]
    Transition:   constant velocity (same as Linear), but with noise modeling.
    Measurement:  [cx, cy] from detections.

    The Kalman filter is more robust than simple polyfit when detections
    are noisy because it maintains an uncertainty estimate and weights
    new observations accordingly.

    BEST FOR: noisy detections, erratic vehicles, occlusion recovery.
    WORST FOR: complex maneuvers (would need higher-order state).
    """

    def __init__(self, horizon: int = 30, process_noise: float = 1.0,
                 measurement_noise: float = 5.0):
        super().__init__(horizon, "KalmanModel")
        self._Q = process_noise       # process noise covariance
        self._R = measurement_noise   # measurement noise covariance

    def predict(self, trail: deque, horizon: Optional[int] = None) -> Optional[MotionPrediction]:
        h = horizon or self.horizon
        trail_list = list(trail)
        if len(trail_list) < 4:
            return None

        # Initialize state from first few observations
        frames, cx_vals, cy_vals = zip(*trail_list)
        cx_arr = np.array(cx_vals, dtype=float)
        cy_arr = np.array(cy_vals, dtype=float)

        # Estimate initial velocity
        vx = float(np.mean(np.diff(cx_arr[-5:]))) if len(cx_arr) >= 5 else float(cx_arr[-1] - cx_arr[-2])
        vy = float(np.mean(np.diff(cy_arr[-5:]))) if len(cy_arr) >= 5 else float(cy_arr[-1] - cy_arr[-2])

        # State: [cx, cy, vx, vy]
        x = np.array([cx_arr[-1], cy_arr[-1], vx, vy])

        # Covariance
        P = np.eye(4) * self._Q

        # Transition matrix (constant velocity)
        F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement matrix (observe position only)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * self._R
        Q = np.eye(4) * self._Q

        # Run Kalman update on last few observations
        for _, cx, cy in trail_list[-min(10, len(trail_list)):]:
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q

            # Update
            z   = np.array([cx, cy])
            y   = z - H @ x
            S   = H @ P @ H.T + R
            K   = P @ H.T @ np.linalg.inv(S)
            x   = x + K @ y
            P   = (np.eye(4) - K @ H) @ P

        # Predict forward
        future = []
        x_pred = x.copy()
        for _ in range(h):
            x_pred = F @ x_pred
            future.append((float(x_pred[0]), float(x_pred[1])))

        vx_final  = float(x[2])
        vy_final  = float(x[3])
        speed     = float(np.sqrt(vx_final**2 + vy_final**2))
        base_conf = 0.85 * min(1.0, len(trail_list) / 15.0)

        return MotionPrediction(
            model_name="KalmanModel",
            future_points=future,
            confidence=self._confidence_decay(base_conf, h),
            vx=vx_final, vy=vy_final, speed=speed,
            metadata={"process_noise": self._Q, "measurement_noise": self._R},
        )


# ==============================================================================
# 5. LSTM MODEL — Neural Network (inference only, requires pretrained weights)
# ==============================================================================

class LSTMModel(BaseMotionModel):
    """
    LSTM-based trajectory prediction.
    Requires a pretrained PyTorch model (trained on NGSIM data).

    Falls back to LinearModel if model file not found or PyTorch not available.

    BEST FOR: lane changes, complex maneuvers, real-world patterns.
    REQUIRES: trained weights at models/motion/lstm_predictor.pt
    """

    def __init__(
        self,
        horizon:      int  = 30,
        model_path:   str  = "models/motion/lstm_predictor.pt",
        seq_len:      int  = 10,
        hidden_size:  int  = 64,
        num_layers:   int  = 2,
    ):
        super().__init__(horizon, "LSTMModel")
        self._model_path   = model_path
        self._seq_len      = seq_len
        self._hidden_size  = hidden_size
        self._num_layers   = num_layers
        self._model        = None
        self._fallback     = LinearModel(horizon=horizon)
        self._load_model()

    def _load_model(self) -> None:
        """Try to load pretrained LSTM weights."""
        try:
            import torch
            from pathlib import Path

            if not Path(self._model_path).exists():
                logger.warning(
                    f"[LSTMModel] Weights not found at {self._model_path}. "
                    f"Using LinearModel as fallback."
                )
                return

            self._model = torch.load(self._model_path, map_location="cpu")
            self._model.eval()
            logger.info(f"[LSTMModel] Loaded pretrained weights from {self._model_path}")

        except ImportError:
            logger.warning("[LSTMModel] PyTorch not available. Using LinearModel fallback.")
        except Exception as e:
            logger.warning(f"[LSTMModel] Failed to load model: {e}. Using fallback.")

    def predict(self, trail: deque, horizon: Optional[int] = None) -> Optional[MotionPrediction]:
        h = horizon or self.horizon

        # Fall back to Linear if model not available
        if self._model is None:
            pred = self._fallback.predict(trail, h)
            if pred:
                pred.model_name = "LSTMModel(fallback→Linear)"
            return pred

        trail_list = list(trail)
        if len(trail_list) < self._seq_len:
            return self._fallback.predict(trail, h)

        try:
            import torch

            # Prepare input sequence
            recent = trail_list[-self._seq_len:]
            _, cx_vals, cy_vals = zip(*recent)
            cx_arr = np.array(cx_vals, dtype=np.float32)
            cy_arr = np.array(cy_vals, dtype=np.float32)

            # Normalize
            cx_mean, cx_std = cx_arr.mean(), cx_arr.std() + 1e-8
            cy_mean, cy_std = cy_arr.mean(), cy_arr.std() + 1e-8
            cx_norm = (cx_arr - cx_mean) / cx_std
            cy_norm = (cy_arr - cy_mean) / cy_std

            x = torch.tensor(
                np.stack([cx_norm, cy_norm], axis=-1),
                dtype=torch.float32
            ).unsqueeze(0)   # (1, seq_len, 2)

            with torch.no_grad():
                output = self._model(x)  # expected: (1, horizon, 2)

            output_np = output.squeeze(0).numpy()  # (horizon, 2)

            future = []
            for t_step in range(min(h, len(output_np))):
                fx = float(output_np[t_step, 0]) * cx_std + cx_mean
                fy = float(output_np[t_step, 1]) * cy_std + cy_mean
                future.append((fx, fy))

            # Pad with linear if output shorter than horizon
            if len(future) < h:
                linear_pred = self._fallback.predict(trail, h - len(future))
                if linear_pred:
                    future.extend(linear_pred.future_points)

            vx = future[1][0] - future[0][0] if len(future) >= 2 else 0.0
            vy = future[1][1] - future[0][1] if len(future) >= 2 else 0.0
            speed = float(np.sqrt(vx**2 + vy**2))

            return MotionPrediction(
                model_name="LSTMModel",
                future_points=future[:h],
                confidence=self._confidence_decay(0.88, h),
                vx=vx, vy=vy, speed=speed,
                metadata={"seq_len": self._seq_len},
            )

        except Exception as e:
            logger.warning(f"[LSTMModel] Inference failed: {e}. Using fallback.")
            pred = self._fallback.predict(trail, h)
            if pred:
                pred.model_name = "LSTMModel(error→Linear)"
            return pred


# ==============================================================================
# AUTO-SELECTOR
# ==============================================================================

class MotionModelSelector:
    """
    Automatically selects the best motion model based on vehicle dynamics.

    SELECTION CRITERIA (checked in order):
      1. speed_variance < 0.20 → Linear (smooth, constant speed)
      2. mean_accel > 0.50     → Acceleration (accelerating/braking)
      3. speed_variance < 0.60 → Polynomial (moderate variation, turning)
      4. Otherwise             → Kalman (noisy/erratic)

    LSTM is only used when explicitly requested (requires pretrained weights).
    """

    def __init__(
        self,
        horizon:              int   = 30,
        variance_threshold:   float = 0.20,
        accel_threshold:      float = 0.50,
        polynomial_threshold: float = 0.60,
        use_lstm:             bool  = False,
        lstm_model_path:      str   = "models/motion/lstm_predictor.pt",
    ):
        self._variance_threshold   = variance_threshold
        self._accel_threshold      = accel_threshold
        self._polynomial_threshold = polynomial_threshold

        self._linear   = LinearModel(horizon=horizon)
        self._accel    = AccelerationModel(horizon=horizon)
        self._poly     = PolynomialModel(horizon=horizon, degree=3)
        self._kalman   = KalmanModel(horizon=horizon)
        self._lstm     = LSTMModel(horizon=horizon, model_path=lstm_model_path) if use_lstm else None

        self._selection_counts: Dict[str, int] = {}
        logger.info(
            f"MotionModelSelector initialized | "
            f"variance_thresh={variance_threshold}, accel_thresh={accel_threshold}"
        )

    def select_and_predict(
        self,
        trail:   deque,
        horizon: Optional[int] = None,
    ) -> Optional[MotionPrediction]:
        """
        Select the best model for this trail and run prediction.

        Args:
            trail:   deque of (frame_idx, cx, cy)
            horizon: Prediction horizon (uses model default if None)

        Returns:
            MotionPrediction from the selected model, or None.
        """
        trail_list = list(trail)
        if len(trail_list) < 3:
            return None

        model = self._select_model(trail_list)
        prediction = model.predict(trail, horizon)

        # Track which model was selected
        name = model.name
        self._selection_counts[name] = self._selection_counts.get(name, 0) + 1

        return prediction

    def _select_model(self, trail_list: list) -> BaseMotionModel:
        """Determine best model from motion statistics."""
        speeds, accels = self._compute_motion_stats(trail_list)

        if not speeds:
            return self._linear

        mean_speed  = float(np.mean(speeds))
        if mean_speed < 0.1:
            return self._linear   # stationary

        # Coefficient of variation
        speed_cv = float(np.std(speeds) / mean_speed) if mean_speed > 0.1 else 0.0

        # Mean absolute acceleration
        mean_accel = float(np.mean(np.abs(accels))) if accels else 0.0

        if speed_cv < self._variance_threshold:
            return self._linear

        if mean_accel > self._accel_threshold:
            return self._accel

        if speed_cv < self._polynomial_threshold:
            return self._poly

        if self._lstm is not None:
            return self._lstm

        return self._kalman

    def _compute_motion_stats(
        self, trail_list: list
    ) -> Tuple[List[float], List[float]]:
        """Compute speed and acceleration time series from trail."""
        if len(trail_list) < 2:
            return [], []

        speeds = []
        accels = []

        for i in range(1, len(trail_list)):
            _, cx1, cy1 = trail_list[i-1]
            _, cx2, cy2 = trail_list[i]
            speed = float(np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2))
            speeds.append(speed)

            if i >= 2:
                prev_speed = speeds[-2]
                accels.append(abs(speed - prev_speed))

        return speeds, accels

    def get_selection_stats(self) -> dict:
        """Return model selection frequency statistics."""
        total = sum(self._selection_counts.values())
        return {
            "total_predictions": total,
            "model_counts":      dict(self._selection_counts),
            "model_fractions": {
                name: round(count / max(total, 1), 4)
                for name, count in self._selection_counts.items()
            },
        }
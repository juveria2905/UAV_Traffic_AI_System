"""
================================================================================
agentic_ai/confidence_scorer.py — Decision Confidence Scoring System
================================================================================

PURPOSE:
  Computes a unified confidence score [0.0, 1.0] for each agent decision
  by combining multiple input signals. This confidence is used to:
  - Filter low-confidence decisions (don't act on uncertain data)
  - Weight decisions in conflict resolution
  - Provide explainable confidence to the dashboard
  - Enable adaptive threshold updates in FeedbackEngine

FORMULA:
  confidence = (
      w_risk      × risk_certainty
    + w_speed     × speed_certainty
    + w_density   × density_certainty
    + w_history   × historical_accuracy
    + w_prediction× prediction_certainty
  )

  Where each certainty is a normalized [0,1] signal.

CONNECTS TO:
  agent_system.py     → Used before every decision to score confidence
  conflict_resolver.py → Higher confidence wins ties
  feedback_engine.py  → Historical accuracy feeds back into confidence
  memory/decision_memory.py → success_rate is the historical_accuracy input
  api.py             → /api/confidence endpoint
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# CONFIDENCE RESULT
# ==============================================================================

@dataclass
class ConfidenceResult:
    """
    Full confidence breakdown for one decision.

    Storing components makes confidence explainable:
    "Confidence is 0.89 because prediction_certainty=0.95, but
     historical_accuracy is only 0.72 (this track was mis-flagged before)."
    """
    track_id:              int
    action:                str
    confidence:            float     # final weighted score [0,1]
    risk_certainty:        float     # how certain is the risk assessment
    speed_certainty:       float     # how certain is the speed measurement
    density_certainty:     float     # how certain is the density estimate
    historical_accuracy:   float     # past decision success rate for this track
    prediction_certainty:  float     # trajectory prediction confidence
    timestamp:             float = field(default_factory=time.time)

    @property
    def confidence_tier(self) -> str:
        """Human-readable confidence tier."""
        if self.confidence >= 0.90:   return "VERY_HIGH"
        if self.confidence >= 0.75:   return "HIGH"
        if self.confidence >= 0.60:   return "MEDIUM"
        if self.confidence >= 0.40:   return "LOW"
        return "VERY_LOW"

    @property
    def should_act(self) -> bool:
        """
        Whether the decision should be executed based on confidence.
        EMERGENCY_STOP acts even at LOW confidence (safety critical).
        Other actions require MEDIUM confidence minimum.
        """
        if self.action == "EMERGENCY_STOP":
            return self.confidence >= 0.30   # very low threshold for safety
        return self.confidence >= 0.50

    def explain(self) -> str:
        """Human-readable breakdown for reasoning chains."""
        return (
            f"Confidence={self.confidence:.2%} [{self.confidence_tier}] | "
            f"risk={self.risk_certainty:.2f} "
            f"speed={self.speed_certainty:.2f} "
            f"density={self.density_certainty:.2f} "
            f"history={self.historical_accuracy:.2f} "
            f"prediction={self.prediction_certainty:.2f}"
        )

    def to_dict(self) -> dict:
        return {
            "track_id":             self.track_id,
            "action":               self.action,
            "confidence":           round(self.confidence, 4),
            "confidence_tier":      self.confidence_tier,
            "should_act":           self.should_act,
            "components": {
                "risk_certainty":       round(self.risk_certainty, 4),
                "speed_certainty":      round(self.speed_certainty, 4),
                "density_certainty":    round(self.density_certainty, 4),
                "historical_accuracy":  round(self.historical_accuracy, 4),
                "prediction_certainty": round(self.prediction_certainty, 4),
            },
            "timestamp": round(self.timestamp, 3),
        }


# ==============================================================================
# CONFIDENCE SCORER
# ==============================================================================

class ConfidenceScorer:
    """
    Computes multi-factor confidence scores for agent decisions.

    DESIGN:
        Each component score is independently normalized to [0, 1].
        Components are then combined via weighted average.
        Weights can be updated by FeedbackEngine based on which
        components were most predictive of correct decisions.

    SCORE COMPONENTS:
        risk_certainty:
            Based on how "clean" the TTC signal is. Very low TTC (<1s)
            with high relative speed = very certain HIGH risk.
            TTC near the threshold (e.g., 2.9s for 3.0s threshold) = uncertain.

        speed_certainty:
            Based on speed variance. If a vehicle has been moving at a
            consistent 8 px/frame, the velocity estimate is reliable.
            If speed jumps between 2 and 15 px/frame, estimates are noisy.

        density_certainty:
            Traffic density estimates have inherent uncertainty.
            Very high density (>0.8) or very low (<0.2) are certain.
            Mid-range density is more uncertain.

        historical_accuracy:
            The track's past success_rate from DecisionMemory.
            If the system consistently made correct decisions about this
            vehicle, use its history to boost confidence.

        prediction_certainty:
            From TrajectoryPredictor.PredictedTrajectory.confidence.
            The predictor itself estimates how reliable its prediction is.
    """

    def __init__(
        self,
        w_risk:       float = 0.30,
        w_speed:      float = 0.20,
        w_density:    float = 0.15,
        w_history:    float = 0.20,
        w_prediction: float = 0.15,
        min_confidence_for_action: float = 0.50,
        emergency_min_confidence:  float = 0.30,
    ):
        """
        Args:
            w_*:        Component weights (normalized internally if don't sum to 1)
            min_confidence_for_action: Minimum to act on non-emergency decisions
            emergency_min_confidence:  Minimum to act on EMERGENCY_STOP
        """
        total = w_risk + w_speed + w_density + w_history + w_prediction
        self._w_risk       = w_risk       / total
        self._w_speed      = w_speed      / total
        self._w_density    = w_density    / total
        self._w_history    = w_history    / total
        self._w_prediction = w_prediction / total

        self._min_conf     = min_confidence_for_action
        self._emerg_min    = emergency_min_confidence

        logger.info(
            f"ConfidenceScorer initialized | "
            f"weights: risk={self._w_risk:.2f}, speed={self._w_speed:.2f}, "
            f"density={self._w_density:.2f}, history={self._w_history:.2f}, "
            f"prediction={self._w_prediction:.2f}"
        )

    def score(
        self,
        track_id:            int,
        action:              str,
        ttc_seconds:         Optional[float] = None,
        ttc_threshold:       float           = 3.0,
        speed:               float           = 0.0,
        speed_variance:      float           = 0.0,
        density:             float           = 0.0,
        historical_accuracy: float           = 0.5,
        prediction_confidence: float         = 0.5,
        collision_risk_score:  float         = 0.0,
    ) -> ConfidenceResult:
        """
        Compute confidence for a specific decision on a specific track.

        Args:
            track_id:             Vehicle track ID
            action:               Proposed decision ("REROUTE", "EMERGENCY_STOP", etc.)
            ttc_seconds:          Current TTC (None if no collision detected)
            ttc_threshold:        TTC threshold for this risk tier
            speed:                Current vehicle speed (px/frame)
            speed_variance:       CV of recent speeds (0=steady, 1=erratic)
            density:              Local traffic density [0,1]
            historical_accuracy:  Track's past success rate from DecisionMemory
            prediction_confidence: Trajectory predictor's own confidence
            collision_risk_score: Overall risk score from CollisionEngine [0,1]

        Returns:
            ConfidenceResult with all component scores and final confidence.
        """
        # ── Component 1: Risk Certainty ──────────────────────────────────────
        risk_certainty = self._compute_risk_certainty(
            ttc_seconds, ttc_threshold, collision_risk_score
        )

        # ── Component 2: Speed Certainty ─────────────────────────────────────
        speed_certainty = self._compute_speed_certainty(speed, speed_variance)

        # ── Component 3: Density Certainty ───────────────────────────────────
        density_certainty = self._compute_density_certainty(density)

        # ── Component 4: Historical Accuracy ─────────────────────────────────
        # Already in [0,1] — trust it directly
        historical_clamped = float(np.clip(historical_accuracy, 0.0, 1.0))

        # ── Component 5: Prediction Certainty ────────────────────────────────
        pred_clamped = float(np.clip(prediction_confidence, 0.0, 1.0))

        # ── Weighted Sum ─────────────────────────────────────────────────────
        confidence = (
            self._w_risk       * risk_certainty
            + self._w_speed    * speed_certainty
            + self._w_density  * density_certainty
            + self._w_history  * historical_clamped
            + self._w_prediction * pred_clamped
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        result = ConfidenceResult(
            track_id=track_id,
            action=action,
            confidence=confidence,
            risk_certainty=risk_certainty,
            speed_certainty=speed_certainty,
            density_certainty=density_certainty,
            historical_accuracy=historical_clamped,
            prediction_certainty=pred_clamped,
        )

        logger.debug(f"[Confidence] Track #{track_id} {action}: {result.explain()}")
        return result

    # ── COMPONENT CALCULATIONS ────────────────────────────────────────────────

    def _compute_risk_certainty(
        self,
        ttc:       Optional[float],
        threshold: float,
        risk_score: float,
    ) -> float:
        """
        Certainty that the risk assessment is correct.

        Logic:
          - No TTC (no collision detected): certainty = 1 - risk_score
            (confident there's no risk IF risk_score is low)
          - TTC very low (< threshold/3): certainty = 0.95 (very confident HIGH)
          - TTC near threshold: certainty lower (close call, borderline case)
          - TTC well above threshold: certainty = 0.90 (confident LOW)
        """
        if ttc is None:
            # No collision detected
            return float(np.clip(1.0 - risk_score, 0.3, 1.0))

        # How far from threshold? Close to threshold = less certain
        distance_from_threshold = abs(ttc - threshold)
        # Normalize: 0 = at threshold (max uncertainty), threshold = far (high certainty)
        closeness = max(0.0, 1.0 - distance_from_threshold / threshold)
        uncertainty = 0.25 * closeness   # max 25% uncertainty at the threshold

        base_certainty = 0.90
        if ttc < threshold / 3.0:
            base_certainty = 0.97   # TTC very low → very certain HIGH risk
        elif ttc < threshold:
            base_certainty = 0.85   # TTC in HIGH zone
        else:
            base_certainty = 0.80   # TTC above threshold (less certain it's risky)

        return float(np.clip(base_certainty - uncertainty, 0.0, 1.0))

    def _compute_speed_certainty(self, speed: float, speed_variance: float) -> float:
        """
        Certainty of speed-based observations.

        Low speed variance → speed is consistent → high certainty.
        High speed variance → speed is noisy → low certainty.
        Also: very low speed → barely moving, uncertainty about direction.
        """
        if speed < 0.5:
            return 0.60   # nearly stationary — uncertain about intent

        # Variance penalty: CV=0 → no penalty, CV=1 → 40% penalty
        variance_penalty = float(np.clip(speed_variance * 0.4, 0.0, 0.40))
        return float(np.clip(0.95 - variance_penalty, 0.50, 0.95))

    def _compute_density_certainty(self, density: float) -> float:
        """
        Certainty of traffic density estimate.

        Extreme values (very high or very low) are certain.
        Mid-range values are less certain (boundary zone).
        """
        # Certainty is high at extremes, lower in middle
        # Shape: high → 1.0, 0.5 → 0.7, 0.0 → 0.9
        # Using a parabolic function: certainty = 0.7 + 0.3*(2*density-1)^2
        certainty = 0.70 + 0.30 * (2 * density - 1.0) ** 2
        return float(np.clip(certainty, 0.60, 1.00))

    # ── WEIGHT ADAPTATION ─────────────────────────────────────────────────────

    def update_weights(self, feedback: Dict[str, float]) -> None:
        """
        Update component weights based on FeedbackEngine feedback.

        Args:
            feedback: Dict of {"w_risk": delta, "w_speed": delta, ...}
                      Positive delta = this component was predictive.
                      Negative delta = this component was misleading.

        Called by FeedbackEngine after evaluating decision outcomes.
        """
        lr = 0.01   # small learning rate for stability
        self._w_risk       = float(np.clip(self._w_risk       + feedback.get("w_risk",       0) * lr, 0.05, 0.60))
        self._w_speed      = float(np.clip(self._w_speed      + feedback.get("w_speed",      0) * lr, 0.05, 0.60))
        self._w_density    = float(np.clip(self._w_density    + feedback.get("w_density",    0) * lr, 0.05, 0.60))
        self._w_history    = float(np.clip(self._w_history    + feedback.get("w_history",    0) * lr, 0.05, 0.60))
        self._w_prediction = float(np.clip(self._w_prediction + feedback.get("w_prediction", 0) * lr, 0.05, 0.60))

        # Re-normalize
        total = self._w_risk + self._w_speed + self._w_density + self._w_history + self._w_prediction
        self._w_risk       /= total
        self._w_speed      /= total
        self._w_density    /= total
        self._w_history    /= total
        self._w_prediction /= total

        logger.debug(
            f"[Confidence] Weights updated: "
            f"risk={self._w_risk:.3f}, speed={self._w_speed:.3f}, "
            f"density={self._w_density:.3f}, history={self._w_history:.3f}, "
            f"prediction={self._w_prediction:.3f}"
        )

    def get_weights(self) -> dict:
        """Return current component weights."""
        return {
            "risk":       round(self._w_risk, 4),
            "speed":      round(self._w_speed, 4),
            "density":    round(self._w_density, 4),
            "history":    round(self._w_history, 4),
            "prediction": round(self._w_prediction, 4),
        }
"""
================================================================================
agentic_ai/priority_engine.py — Multi-Factor Priority Scoring Engine
================================================================================

PURPOSE:
  Computes a composite priority score for each tracked vehicle that drives
  agent decision-making. Higher score = needs attention first.

FORMULA:
  priority_score = 0.35 × risk_score
                 + 0.25 × speed_variance_factor
                 + 0.20 × traffic_density_factor
                 + 0.20 × historical_performance_factor

WHY MULTI-FACTOR:
  Using only TTC (Time-to-Collision) would miss vehicles that are:
  - Driving erratically (high speed variance) but currently not on collision path
  - In dense traffic zones where emergent collisions are more likely
  - Repeat offenders (historically flagged vehicles deserve more attention)

CONNECTS TO:
  - agent_system.py    → CoordinatorAgent uses priority scores for decision ordering
  - hierarchy_manager.py → ZoneAgent uses scores for workload distribution
  - conflict_resolver.py → Higher score wins conflicts between competing decisions
  - reasoning_chain.py → Priority score is logged as a reasoning step
================================================================================
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils.logger import get_logger, ModuleStats

logger = get_logger(__name__)


# ==============================================================================
# PRIORITY RESULT
# ==============================================================================

@dataclass
class PriorityScore:
    """
    Full priority breakdown for one tracked vehicle.

    Storing the component scores (not just the final score) enables:
    - Explainable AI: "High priority because high speed variance (0.83)"
    - Dashboard visualization: stacked bar chart of score components
    - Debugging: understand exactly why a vehicle was prioritized
    """
    track_id:                  int
    frame_idx:                 int
    final_score:               float   # 0.0 – 1.0 (higher = more urgent)
    risk_component:            float   # from collision engine
    speed_variance_component:  float   # erratic movement
    density_component:         float   # local traffic density
    historical_component:      float   # past behavior
    confidence:                float   # how reliable this score is
    timestamp:                 float   = field(default_factory=time.time)

    @property
    def priority_level(self) -> str:
        """Human-readable priority tier."""
        if self.final_score >= 0.80:   return "CRITICAL"
        if self.final_score >= 0.60:   return "HIGH"
        if self.final_score >= 0.40:   return "MEDIUM"
        if self.final_score >= 0.20:   return "LOW"
        return "NOMINAL"

    def to_dict(self) -> dict:
        return {
            "track_id":         self.track_id,
            "frame_idx":        self.frame_idx,
            "final_score":      round(self.final_score, 4),
            "priority_level":   self.priority_level,
            "risk_component":   round(self.risk_component, 4),
            "speed_variance":   round(self.speed_variance_component, 4),
            "density":          round(self.density_component, 4),
            "historical":       round(self.historical_component, 4),
            "confidence":       round(self.confidence, 4),
            "timestamp":        round(self.timestamp, 3),
        }

    def explain(self) -> str:
        """
        Generate a human-readable explanation string for reasoning chains
        and dashboard display.

        Example:
            "Track #7: CRITICAL (0.91) | risk=0.95×0.35 + variance=0.80×0.25 + density=0.70×0.20 + history=0.85×0.20"
        """
        return (
            f"Track #{self.track_id}: {self.priority_level} "
            f"(score={self.final_score:.3f}) | "
            f"risk={self.risk_component:.2f}×0.35 + "
            f"variance={self.speed_variance_component:.2f}×0.25 + "
            f"density={self.density_component:.2f}×0.20 + "
            f"history={self.historical_component:.2f}×0.20"
        )


# ==============================================================================
# PRIORITY ENGINE
# ==============================================================================

class PriorityEngine:
    """
    Computes multi-factor priority scores for all active tracks.

    ADAPTIVE WEIGHTS:
        The default weights (0.35, 0.25, 0.20, 0.20) can adapt based on
        the scenario:
        - Dense urban airspace → increase density weight
        - High-speed corridor  → increase speed variance weight
        - Known risk zones     → increase historical weight

    SPEED VARIANCE CALCULATION:
        variance = std(speeds over last N frames) / mean(speeds)
        This is the Coefficient of Variation (CV) — measures how erratic
        a vehicle's speed is regardless of its absolute speed.
        CV=0.0 → perfectly constant speed (predictable)
        CV=1.0 → highly erratic speed (unpredictable)

    DENSITY CALCULATION:
        density = num_vehicles_within_radius / max_expected_vehicles
        Normalized to [0, 1]. High density = more emergent collision risk.

    HISTORICAL PERFORMANCE:
        Tracks that have been flagged HIGH risk before get higher historical
        scores. This prevents the system from ignoring a vehicle that was
        briefly rerouted and is now approaching again.
    """

    def __init__(
        self,
        weight_risk:       float = 0.35,
        weight_variance:   float = 0.25,
        weight_density:    float = 0.20,
        weight_historical: float = 0.20,
        speed_history_len: int   = 15,    # frames of speed history for variance
        density_radius_px: float = 200.0,  # radius to count nearby vehicles
        max_density_count: int   = 20,     # maximum expected vehicles in radius
        enable_adaptive:   bool  = True,   # allow weights to adapt
        adaptive_lr:       float = 0.01,   # learning rate for weight updates
    ):
        """
        Args:
            weight_*:       Component weights (must sum to 1.0).
            speed_history_len: How many frames of speed history to keep per track.
            density_radius_px: Radius in pixels to count neighboring vehicles.
            max_density_count: Normalizes density factor (count/max_density_count).
            enable_adaptive:   If True, weights shift based on scenario patterns.
            adaptive_lr:       Learning rate for adaptive weight updates.
        """
        # Validate weights sum to 1.0
        total = weight_risk + weight_variance + weight_density + weight_historical
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"PriorityEngine weights sum to {total:.3f}, not 1.0. "
                f"Normalizing automatically."
            )
            weight_risk       /= total
            weight_variance   /= total
            weight_density    /= total
            weight_historical /= total

        self._w_risk       = weight_risk
        self._w_variance   = weight_variance
        self._w_density    = weight_density
        self._w_historical = weight_historical

        self._density_radius   = density_radius_px
        self._max_density      = max_density_count
        self._speed_hist_len   = speed_history_len
        self._enable_adaptive  = enable_adaptive
        self._adaptive_lr      = adaptive_lr

        # Per-track speed history for variance calculation
        # {track_id: deque of (frame_idx, speed_px_per_frame)}
        self._speed_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=speed_history_len)
        )

        # Per-track risk history for historical component
        # {track_id: list of (frame_idx, risk_score)}
        self._risk_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )

        # All-time priority scores per track (for trend analysis)
        self._score_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        self._stats = ModuleStats("PriorityEngine")
        logger.info(
            f"PriorityEngine initialized | "
            f"weights: risk={self._w_risk}, var={self._w_variance}, "
            f"density={self._w_density}, hist={self._w_historical}"
        )

    # ── MAIN SCORING FUNCTION ─────────────────────────────────────────────────

    def compute_scores(
        self,
        frame_idx:     int,
        tracks:        dict,
        risk_scores:   Optional[Dict[int, float]] = None,
        collision_events: Optional[list]          = None,
    ) -> Dict[int, PriorityScore]:
        """
        Compute priority scores for all active tracks.

        Args:
            frame_idx:        Current frame number
            tracks:           Dict of {track_id: TrackedObject}
            risk_scores:      Optional pre-computed risk scores per track
                              (e.g., from CollisionEngine). If None, derived
                              from collision_events.
            collision_events: List of CollisionEvent objects from CollisionEngine.
                              Used to build risk_scores if not provided directly.

        Returns:
            Dict of {track_id: PriorityScore} for all tracks.
        """
        t_start = time.perf_counter()

        # Build risk score lookup from collision events
        if risk_scores is None:
            risk_scores = self._build_risk_scores(collision_events or [])

        # Compute density map once (shared across all track scoring)
        density_map = self._compute_density_map(tracks)

        scores: Dict[int, PriorityScore] = {}

        for track_id, track in tracks.items():
            score = self._score_track(
                track_id=track_id,
                track=track,
                frame_idx=frame_idx,
                risk_score=risk_scores.get(track_id, 0.0),
                density_factor=density_map.get(track_id, 0.0),
            )
            scores[track_id] = score

            # Store for history
            self._score_history[track_id].append((frame_idx, score.final_score))
            self._risk_history[track_id].append((frame_idx, score.risk_component))

        # Adaptive weight update (if enabled)
        if self._enable_adaptive:
            self._adapt_weights(scores)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._stats.record(elapsed_ms, success=True)

        logger.debug(
            f"[PriorityEngine] Frame {frame_idx}: "
            f"scored {len(scores)} tracks in {elapsed_ms:.1f}ms"
        )

        return scores

    # ── COMPONENT CALCULATIONS ────────────────────────────────────────────────

    def _score_track(
        self,
        track_id:      int,
        track,
        frame_idx:     int,
        risk_score:    float,
        density_factor: float,
    ) -> PriorityScore:
        """
        Compute the full PriorityScore for a single track.

        All component scores are in [0, 1].
        """
        # ── Component 1: Risk Score ──────────────────────────────────────────
        # Passed in directly from CollisionEngine (TTC-based)
        risk_component = float(np.clip(risk_score, 0.0, 1.0))

        # ── Component 2: Speed Variance ──────────────────────────────────────
        speed_variance_component = self._compute_speed_variance(track_id, track, frame_idx)

        # ── Component 3: Traffic Density ─────────────────────────────────────
        density_component = float(np.clip(density_factor, 0.0, 1.0))

        # ── Component 4: Historical Performance ──────────────────────────────
        historical_component = self._compute_historical(track_id)

        # ── Weighted Sum ─────────────────────────────────────────────────────
        final_score = (
            self._w_risk       * risk_component
            + self._w_variance * speed_variance_component
            + self._w_density  * density_component
            + self._w_historical * historical_component
        )
        final_score = float(np.clip(final_score, 0.0, 1.0))

        # ── Confidence ───────────────────────────────────────────────────────
        # Confidence is lower when:
        # - Track is young (short history → unreliable variance estimate)
        # - Speed history is sparse
        trail_len   = len(track.trail) if hasattr(track, "trail") else 0
        spd_hist    = len(self._speed_history[track_id])
        confidence  = float(np.clip(
            0.5 * (trail_len / 20.0) + 0.5 * (spd_hist / self._speed_hist_len),
            0.1, 1.0
        ))

        return PriorityScore(
            track_id=track_id,
            frame_idx=frame_idx,
            final_score=final_score,
            risk_component=risk_component,
            speed_variance_component=speed_variance_component,
            density_component=density_component,
            historical_component=historical_component,
            confidence=confidence,
        )

    def _compute_speed_variance(self, track_id: int, track, frame_idx: int) -> float:
        """
        Compute normalized speed variance (Coefficient of Variation).

        WHY COEFFICIENT OF VARIATION (CV):
            CV = std / mean
            CV is scale-independent: a vehicle moving at 100px/frame with
            std=10 has the same CV as one at 10px/frame with std=1.
            This correctly captures "erratic behavior" regardless of speed.

        Returns:
            float in [0, 1]: 0 = steady speed, 1 = highly erratic
        """
        # Get current speed
        vel = track.velocity if hasattr(track, "velocity") else None
        if vel is not None and vel:
            vx, vy = vel
            speed = float(np.sqrt(vx**2 + vy**2))
        else:
            speed = 0.0

        self._speed_history[track_id].append((frame_idx, speed))

        hist = self._speed_history[track_id]
        if len(hist) < 3:
            return 0.0   # insufficient history

        speeds = np.array([s for _, s in hist])
        mean_speed = np.mean(speeds)
        if mean_speed < 0.1:
            return 0.0   # essentially stationary — no meaningful variance

        cv = np.std(speeds) / mean_speed   # coefficient of variation
        # Clip at 2.0 (CV > 2.0 is extremely erratic) and normalize to [0, 1]
        return float(np.clip(cv / 2.0, 0.0, 1.0))

    def _compute_density_map(self, tracks: dict) -> Dict[int, float]:
        """
        Compute local traffic density factor for each track.

        For each vehicle, count how many other vehicles are within
        density_radius_px. Normalize by max_density_count.

        WHY SPATIAL DENSITY MATTERS:
            In dense traffic, even vehicles not currently on collision paths
            have higher emergent collision probability. Dense zones need
            more proactive management.

        O(N²) computation — acceptable for N < 200 vehicles.
        For N > 200, use spatial indexing (k-d tree).

        Returns:
            {track_id: density_factor} where density_factor ∈ [0, 1]
        """
        density_map: Dict[int, float] = {}
        track_ids    = list(tracks.keys())
        centers      = {}

        for tid, track in tracks.items():
            c = track.center if hasattr(track, "center") else None
            if c:
                centers[tid] = np.array(c)

        for tid in track_ids:
            if tid not in centers:
                density_map[tid] = 0.0
                continue

            pos   = centers[tid]
            count = 0
            for other_tid, other_pos in centers.items():
                if other_tid == tid:
                    continue
                dist = float(np.linalg.norm(pos - other_pos))
                if dist <= self._density_radius:
                    count += 1

            density_map[tid] = float(np.clip(count / self._max_density, 0.0, 1.0))

        return density_map

    def _compute_historical(self, track_id: int) -> float:
        """
        Compute historical risk factor for a track.

        Looks at the track's recent risk history and returns a weighted
        average biased toward recent events (more recent = more weight).

        A track that was HIGH risk 3 frames ago is more concerning than
        one that was HIGH risk 30 frames ago.

        Returns:
            float in [0, 1]: 0 = consistently low risk, 1 = consistently high risk
        """
        hist = self._risk_history[track_id]
        if not hist:
            return 0.0

        risks   = np.array([r for _, r in hist])
        n       = len(risks)
        # Exponential recency weighting: most recent has weight 1.0
        weights = np.exp(np.linspace(-2.0, 0.0, n))
        weights /= weights.sum()

        return float(np.clip(np.dot(risks, weights), 0.0, 1.0))

    # ── ADAPTIVE WEIGHTS ──────────────────────────────────────────────────────

    def _adapt_weights(self, scores: Dict[int, PriorityScore]) -> None:
        """
        Adaptively shift weights based on current scenario patterns.

        LOGIC:
            If average risk score across vehicles is very high (>0.7):
                → Increase risk weight (collision avoidance is the priority)
            If average speed variance is very high (>0.6):
                → Increase speed variance weight (erratic traffic)
            After shifts: re-normalize so weights still sum to 1.0

        This is NOT machine learning — it's rule-based adaptation.
        For true ML adaptation, see learning/feedback_engine.py.
        """
        if not scores:
            return

        avg_risk     = np.mean([s.risk_component            for s in scores.values()])
        avg_variance = np.mean([s.speed_variance_component  for s in scores.values()])

        lr = self._adaptive_lr

        # Increase risk weight when risk is high
        if avg_risk > 0.7:
            self._w_risk       += lr
            self._w_historical -= lr / 3
            self._w_density    -= lr / 3
            self._w_variance   -= lr / 3

        # Increase variance weight when traffic is erratic
        if avg_variance > 0.6:
            self._w_variance   += lr
            self._w_historical -= lr / 3
            self._w_density    -= lr / 3
            self._w_risk       -= lr / 3

        # Re-normalize
        total = self._w_risk + self._w_variance + self._w_density + self._w_historical
        self._w_risk       /= total
        self._w_variance   /= total
        self._w_density    /= total
        self._w_historical /= total

        # Clip to prevent any weight from going negative or above 0.8
        self._w_risk       = float(np.clip(self._w_risk,       0.05, 0.80))
        self._w_variance   = float(np.clip(self._w_variance,   0.05, 0.80))
        self._w_density    = float(np.clip(self._w_density,    0.05, 0.80))
        self._w_historical = float(np.clip(self._w_historical, 0.05, 0.80))

    # ── UTILITY ───────────────────────────────────────────────────────────────

    def _build_risk_scores(self, collision_events: list) -> Dict[int, float]:
        """
        Convert CollisionEngine events to per-track risk scores [0, 1].

        Uses the maximum risk across all events involving each track.
        HIGH   → 0.9 – 1.0
        MEDIUM → 0.5 – 0.7
        LOW    → 0.2 – 0.4
        """
        risk_scores: Dict[int, float] = defaultdict(float)
        risk_map = {"HIGH": 0.95, "MEDIUM": 0.60, "LOW": 0.25, "NONE": 0.0}

        for event in collision_events:
            risk_val = risk_map.get(
                event.risk_level.value if hasattr(event.risk_level, "value")
                else str(event.risk_level),
                0.0
            )
            # Additional precision: use TTC to scale within the tier
            ttc = getattr(event, "ttc_seconds", 10.0)
            ttc_factor = max(0.0, 1.0 - ttc / 12.0)   # 0s TTC → 1.0, 12s → 0.0

            final_risk = 0.7 * risk_val + 0.3 * ttc_factor

            # Apply to both vehicles in the collision pair
            for tid in [event.track_id_a, event.track_id_b]:
                risk_scores[tid] = max(risk_scores[tid], float(np.clip(final_risk, 0.0, 1.0)))

        return dict(risk_scores)

    def get_top_priority_tracks(
        self,
        scores: Dict[int, PriorityScore],
        n: int = 5,
        min_score: float = 0.3,
    ) -> List[PriorityScore]:
        """
        Return the top-N highest priority tracks.

        Args:
            scores:    Output from compute_scores()
            n:         Maximum number of tracks to return
            min_score: Only return tracks above this threshold

        Used by CoordinatorAgent to focus attention on most urgent vehicles.
        """
        filtered = [s for s in scores.values() if s.final_score >= min_score]
        return sorted(filtered, key=lambda s: s.final_score, reverse=True)[:n]

    def get_current_weights(self) -> dict:
        """Return current (possibly adapted) weights for dashboard display."""
        return {
            "risk":       round(self._w_risk,       4),
            "variance":   round(self._w_variance,   4),
            "density":    round(self._w_density,    4),
            "historical": round(self._w_historical, 4),
        }

    def get_stats_summary(self) -> dict:
        return self._stats.summary()
"""
================================================================================
agentic_ai/learning/feedback_engine.py — Adaptive Feedback Learning Engine
================================================================================

PURPOSE:
  Observes the outcomes of past decisions and feeds reward/penalty signals
  back into the system to improve future decisions. This closes the
  decision → action → outcome → adaptation loop.

FEEDBACK LOOP:
  Frame N:   Decision = REROUTE for Track #7 (TTC=2.1s)
  Frame N+5: Track #7 TTC improved to 4.5s → reward = +0.8
             (reroute was effective — vehicle moved away from danger)

  Frame M:   Decision = EMERGENCY_STOP for Track #3 (TTC=2.8s)
  Frame M+5: Track #3 TTC still 2.9s (barely changed) → reward = -0.2
             (emergency stop was probably unnecessary at TTC=2.8s)

WHAT GETS UPDATED:
  - DecisionMemory: outcome_score for past decisions
  - ConfidenceScorer: weights adjusted based on what was predictive
  - PriorityEngine: adaptive weights shifted based on outcomes
  - GoalPlanner: goal scores updated

CONNECTS TO:
  agent_system.py      → FeedbackEngine.evaluate() called each frame
  memory/decision_memory.py → Updates outcome scores
  confidence_scorer.py → update_weights() called with feedback deltas
  goal_planner.py      → update_goal_score() called with performance
================================================================================
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackRecord:
    """One feedback evaluation (decision → observed outcome)."""
    track_id:      int
    frame_idx:     int
    action:        str
    reward:        float    # -1.0 to +1.0
    ttc_before:    float    # TTC when decision was made
    ttc_after:     float    # TTC N frames later
    explanation:   str
    timestamp:     float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "track_id":    self.track_id,
            "frame_idx":   self.frame_idx,
            "action":      self.action,
            "reward":      round(self.reward, 4),
            "ttc_before":  round(self.ttc_before, 3),
            "ttc_after":   round(self.ttc_after, 3),
            "improvement": round(self.ttc_after - self.ttc_before, 3),
            "explanation": self.explanation,
            "timestamp":   round(self.timestamp, 3),
        }


class FeedbackEngine:
    """
    Evaluates past decisions by comparing the situation before and after.
    Issues reward/penalty signals to improve future decision-making.

    EVALUATION WINDOW:
        We evaluate decisions made `lookback_frames` ago by comparing
        the TTC then vs now. If TTC improved significantly → reward.
        If TTC worsened or didn't change → penalty.

    REWARD FUNCTION (for REROUTE / EMERGENCY_STOP):
        reward = clip((ttc_after - ttc_before) / ttc_before, -1.0, +1.0)
        Positive = situation improved (TTC increased)
        Negative = situation worsened (TTC decreased)

    For HOLD: reward based on density reduction.
    For MONITOR: reward = +0.1 if nothing bad happened (correct to monitor).
    For EMERGENCY_STOP: reward based on whether TTC was genuinely critical.
    """

    def __init__(
        self,
        lookback_frames:  int   = 10,
        min_ttc_improvement: float = 1.0,   # seconds of TTC improvement = good
        history_maxlen:   int   = 500,
        enable_learning:  bool  = True,
    ):
        self._lookback         = lookback_frames
        self._min_improvement  = min_ttc_improvement
        self._enable_learning  = enable_learning
        self._history:         deque[FeedbackRecord] = deque(maxlen=history_maxlen)
        self._reward_totals:   Dict[str, float] = defaultdict(float)
        self._reward_counts:   Dict[str, int]   = defaultdict(int)

        # Pending evaluations: {(track_id, frame_idx): {action, ttc_before, ...}}
        self._pending: Dict[Tuple[int,int], dict] = {}

        logger.info(
            f"FeedbackEngine initialized | "
            f"lookback={lookback_frames} frames, learning={enable_learning}"
        )

    def register_decision(
        self,
        track_id:      int,
        frame_idx:     int,
        action:        str,
        ttc_at_time:   float,
        density:       float = 0.0,
        confidence:    float = 0.5,
    ) -> None:
        """
        Register a decision for future evaluation.
        Called by agent_system.py each time a decision is made.

        Args:
            track_id:     Vehicle track ID
            frame_idx:    Frame when decision was made
            action:       Decision type ("REROUTE", "EMERGENCY_STOP", etc.)
            ttc_at_time:  TTC in seconds at decision time
            density:      Traffic density at decision time
            confidence:   Decision confidence
        """
        key = (track_id, frame_idx)
        self._pending[key] = {
            "action":       action,
            "ttc_before":   ttc_at_time,
            "density":      density,
            "confidence":   confidence,
            "registered_at": frame_idx,
        }

    def evaluate(
        self,
        current_frame:  int,
        current_ttcs:   Dict[int, float],  # {track_id: current_ttc}
        current_densities: Dict[int, float] = None,
    ) -> List[FeedbackRecord]:
        """
        Evaluate pending decisions whose lookback window has elapsed.
        Returns list of new FeedbackRecords generated this frame.

        Args:
            current_frame:  Current frame index
            current_ttcs:   Current TTC for each active track
            current_densities: Current density for each track (optional)

        Called every frame from agent_system.py.
        """
        densities = current_densities or {}
        new_feedback: List[FeedbackRecord] = []
        to_remove = []

        for (track_id, decision_frame), info in self._pending.items():
            # Check if lookback window has elapsed
            frames_elapsed = current_frame - decision_frame
            if frames_elapsed < self._lookback:
                continue

            # Window elapsed — evaluate
            to_remove.append((track_id, decision_frame))

            ttc_before = info["ttc_before"]
            ttc_after  = current_ttcs.get(track_id, ttc_before + 5.0)
            action     = info["action"]

            reward, explanation = self._compute_reward(
                action=action,
                ttc_before=ttc_before,
                ttc_after=ttc_after,
                density_before=info.get("density", 0.0),
                density_after=densities.get(track_id, 0.0),
                confidence=info.get("confidence", 0.5),
            )

            record = FeedbackRecord(
                track_id=track_id,
                frame_idx=decision_frame,
                action=action,
                reward=reward,
                ttc_before=ttc_before,
                ttc_after=ttc_after,
                explanation=explanation,
            )
            self._history.append(record)
            self._reward_totals[action] += reward
            self._reward_counts[action] += 1
            new_feedback.append(record)

        for key in to_remove:
            del self._pending[key]

        return new_feedback

    def _compute_reward(
        self,
        action:         str,
        ttc_before:     float,
        ttc_after:      float,
        density_before: float,
        density_after:  float,
        confidence:     float,
    ) -> Tuple[float, str]:
        """
        Compute reward for a past decision.

        REWARD LOGIC:
          EMERGENCY_STOP: Good if TTC was genuinely < 2s.
                          If TTC was > 3s and unchanged → penalty (false alarm).
          REROUTE:        Good if TTC improved (increased) by >1s.
          HOLD:           Good if density decreased in the zone.
          MONITOR:        Good if nothing bad happened (TTC stayed > 4s).
          PRIORITIZE:     Harder to evaluate — reward = +0.1 if no emergency followed.

        Returns:
            (reward, explanation)
        """
        if action == "EMERGENCY_STOP":
            if ttc_before < 2.0:
                reward = +0.80
                expl = f"CORRECT emergency: TTC={ttc_before:.1f}s was genuinely critical"
            elif ttc_before < 3.0:
                # Borderline — evaluate by whether situation improved
                improvement = (ttc_after - ttc_before) / max(ttc_before, 0.1)
                reward = float(np.clip(improvement * 0.5, -0.3, 0.5))
                expl = f"Borderline emergency: TTC={ttc_before:.1f}s, reward={reward:.2f}"
            else:
                reward = -0.30
                expl = f"FALSE alarm: TTC={ttc_before:.1f}s was not critical enough for emergency"

        elif action == "REROUTE":
            improvement = ttc_after - ttc_before
            if improvement >= self._min_improvement:
                reward = float(np.clip(improvement / 5.0, 0.0, +1.0))
                expl = f"EFFECTIVE reroute: TTC improved +{improvement:.1f}s"
            elif improvement >= 0:
                reward = +0.10
                expl = f"Marginal reroute: TTC improved +{improvement:.1f}s (small)"
            else:
                reward = float(np.clip(improvement / 5.0, -0.5, 0.0))
                expl = f"INEFFECTIVE reroute: TTC worsened by {abs(improvement):.1f}s"

        elif action == "HOLD":
            density_improvement = density_before - density_after
            if density_improvement > 0.1:
                reward = float(np.clip(density_improvement * 2, 0.0, +0.8))
                expl = f"EFFECTIVE hold: density reduced by {density_improvement:.2f}"
            elif density_improvement >= 0:
                reward = +0.05
                expl = "Marginal hold: slight density improvement"
            else:
                reward = -0.10
                expl = f"UNNECESSARY hold: density increased by {abs(density_improvement):.2f}"

        elif action == "MONITOR":
            if ttc_after >= 4.0:
                reward = +0.15
                expl = f"CORRECT monitor: TTC={ttc_after:.1f}s — situation safe"
            elif ttc_after < 2.0:
                reward = -0.40
                expl = f"MISSED risk: track became HIGH risk (TTC={ttc_after:.1f}s) after monitoring"
            else:
                reward = 0.0
                expl = "Neutral monitor: situation unchanged"

        elif action == "PRIORITIZE":
            # Prioritize is hard to evaluate directly; slight positive if no emergency
            reward = +0.05
            expl = "Priority maintenance — indirect benefit"

        else:
            reward = 0.0
            expl = f"Unknown action '{action}' — no evaluation"

        return float(np.clip(reward, -1.0, +1.0)), expl

    def get_average_reward(self, action: Optional[str] = None) -> float:
        """Get average reward for a specific action type (or overall)."""
        if action:
            count = self._reward_counts.get(action, 0)
            total = self._reward_totals.get(action, 0.0)
            return total / count if count > 0 else 0.0

        total = sum(self._reward_totals.values())
        count = sum(self._reward_counts.values())
        return total / count if count > 0 else 0.0

    def get_reward_summary(self) -> dict:
        """Return reward statistics per action type."""
        summary = {}
        for action in self._reward_counts:
            count = self._reward_counts[action]
            total = self._reward_totals[action]
            summary[action] = {
                "count":       count,
                "avg_reward":  round(total / count if count > 0 else 0.0, 4),
                "total":       round(total, 4),
            }
        return summary

    def get_recent_feedback(self, n: int = 20) -> List[dict]:
        """Return most recent feedback records for dashboard."""
        recent = list(reversed(list(self._history)))[:n]
        return [r.to_dict() for r in recent]

    def get_confidence_weight_deltas(self) -> Dict[str, float]:
        """
        Compute suggested weight adjustments for ConfidenceScorer.
        Based on which signals were most correlated with high rewards.

        Simplified heuristic: if REROUTE is consistently high-reward,
        the risk and prediction signals (which drive REROUTE) are reliable.
        """
        reroute_reward  = self.get_average_reward("REROUTE")
        emerg_reward    = self.get_average_reward("EMERGENCY_STOP")
        monitor_reward  = self.get_average_reward("MONITOR")

        return {
            "w_risk":       +0.01 if emerg_reward > 0.5 else (-0.01 if emerg_reward < -0.1 else 0.0),
            "w_prediction": +0.01 if reroute_reward > 0.4 else 0.0,
            "w_history":    +0.01 if monitor_reward > 0.1 else 0.0,
            "w_speed":      0.0,
            "w_density":    0.0,
        }

    def get_stats(self) -> dict:
        return {
            "total_evaluated":   sum(self._reward_counts.values()),
            "pending_count":     len(self._pending),
            "overall_avg_reward": round(self.get_average_reward(), 4),
            "per_action":        self.get_reward_summary(),
        }
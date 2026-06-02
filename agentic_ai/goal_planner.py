"""
================================================================================
agentic_ai/goal_planner.py — Goal-Driven Planning System
================================================================================

PURPOSE:
  Maintains system-level goals and translates them into decision biases
  that influence agent behavior over time. Instead of purely reactive
  (respond to current TTC), the system also acts with intention:
  "Currently prioritizing: minimize congestion in Zone-Z03"

GOALS:
  MINIMIZE_COLLISIONS   → Weight emergency/reroute decisions higher
  MINIMIZE_CONGESTION   → Bias HOLD decisions in dense zones
  MAINTAIN_FLOW         → Prefer REROUTE over HOLD (keep traffic moving)
  PRIORITIZE_EMERGENCY  → Reserve fast corridors for emergency vehicles
  MAXIMIZE_THROUGHPUT   → Minimize unnecessary holds, prefer monitoring

ARCHITECTURE:
  GoalPlanner holds a weighted goal stack.
  Each goal produces a decision_bias dict: {action → additive score weight}
  PriorityEngine and ConflictResolver receive these biases to skew decisions.

CONNECTS TO:
  agent_system.py       → HierarchicalAgentSystem calls planner.get_biases()
  conflict_resolver.py  → Biases influence weighted vote
  priority_engine.py    → Goal biases shift adaptive weights
  feedback_engine.py    → Updates goal weights based on outcome
  api.py               → /api/goals endpoint
  dashboard/app.py      → "Agent Intelligence" tab shows active goals
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# GOAL DEFINITIONS
# ==============================================================================

class Goal(Enum):
    """System-level operational goals."""
    MINIMIZE_COLLISIONS   = "MINIMIZE_COLLISIONS"
    MINIMIZE_CONGESTION   = "MINIMIZE_CONGESTION"
    MAINTAIN_FLOW         = "MAINTAIN_FLOW"
    PRIORITIZE_EMERGENCY  = "PRIORITIZE_EMERGENCY"
    MAXIMIZE_THROUGHPUT   = "MAXIMIZE_THROUGHPUT"


# Decision biases per goal:
# Positive value = prefer this action type
# Negative value = avoid this action type
GOAL_BIASES: Dict[Goal, Dict[str, float]] = {
    Goal.MINIMIZE_COLLISIONS: {
        "EMERGENCY_STOP": +0.20,
        "REROUTE":        +0.15,
        "HOLD":           +0.05,
        "MONITOR":        -0.10,
        "PRIORITIZE":     +0.00,
    },
    Goal.MINIMIZE_CONGESTION: {
        "EMERGENCY_STOP": +0.00,
        "REROUTE":        +0.10,
        "HOLD":           +0.15,
        "MONITOR":        +0.05,
        "PRIORITIZE":     -0.05,
    },
    Goal.MAINTAIN_FLOW: {
        "EMERGENCY_STOP": -0.05,
        "REROUTE":        +0.15,
        "HOLD":           -0.10,
        "MONITOR":        +0.10,
        "PRIORITIZE":     +0.05,
    },
    Goal.PRIORITIZE_EMERGENCY: {
        "EMERGENCY_STOP": +0.25,
        "REROUTE":        +0.05,
        "HOLD":           -0.05,
        "MONITOR":        -0.10,
        "PRIORITIZE":     +0.15,
    },
    Goal.MAXIMIZE_THROUGHPUT: {
        "EMERGENCY_STOP": -0.10,
        "REROUTE":        +0.05,
        "HOLD":           -0.15,
        "MONITOR":        +0.20,
        "PRIORITIZE":     +0.00,
    },
}


@dataclass
class GoalState:
    """State of one active goal."""
    goal:       Goal
    weight:     float     = 1.0    # importance weight [0, 2]
    active:     bool      = True
    activated_at: float   = field(default_factory=time.time)
    reason:     str       = ""     # why this goal is active
    score:      float     = 0.0    # current performance score [0, 1]

    def to_dict(self) -> dict:
        return {
            "goal":         self.goal.value,
            "weight":       round(self.weight, 4),
            "active":       self.active,
            "reason":       self.reason,
            "score":        round(self.score, 4),
            "activated_at": round(self.activated_at, 3),
        }


# ==============================================================================
# GOAL PLANNER
# ==============================================================================

class GoalPlanner:
    """
    Maintains and evaluates system-level goals, producing decision biases.

    USAGE:
        planner = GoalPlanner()

        # Update from current system state
        planner.update(
            collision_count=3,
            congestion_zones=["Z03", "Z07"],
            avg_throughput=12.5,
            emergency_detected=False,
        )

        # Get biases for conflict resolver
        biases = planner.get_action_biases()
        # → {"EMERGENCY_STOP": 0.12, "REROUTE": 0.08, ...}

        # Get active goal list for dashboard
        goals = planner.get_active_goals()
    """

    def __init__(
        self,
        default_goals: Optional[List[Goal]] = None,
        adaptation_rate: float = 0.05,
    ):
        """
        Args:
            default_goals:    Initial active goals (default: all except PRIORITIZE_EMERGENCY)
            adaptation_rate:  How fast goal weights shift in response to outcomes
        """
        self._adaptation_rate = adaptation_rate
        self._goal_states: Dict[Goal, GoalState] = {}

        defaults = default_goals or [
            Goal.MINIMIZE_COLLISIONS,
            Goal.MAINTAIN_FLOW,
            Goal.MINIMIZE_CONGESTION,
            Goal.MAXIMIZE_THROUGHPUT,
        ]
        for g in defaults:
            self._goal_states[g] = GoalState(
                goal=g,
                weight=1.0,
                active=True,
                reason="Default system goal",
            )

        logger.info(
            f"GoalPlanner initialized with goals: "
            f"{[g.value for g in defaults]}"
        )

    def update(
        self,
        collision_count:    int   = 0,
        congestion_zones:   List[str] = None,
        avg_throughput:     float = 0.0,
        emergency_detected: bool  = False,
        frame_idx:          int   = 0,
    ) -> None:
        """
        Update goal relevance based on current system state.
        Automatically activates/deactivates goals and adjusts weights.

        Args:
            collision_count:    Number of active collision risks this frame
            congestion_zones:   List of congested zone IDs
            avg_throughput:     Average vehicles processed per second
            emergency_detected: Whether any EMERGENCY_STOP was issued
            frame_idx:          Current frame number
        """
        congested = congestion_zones or []

        # ── Rule: Boost MINIMIZE_COLLISIONS if collisions are high ────────────
        if Goal.MINIMIZE_COLLISIONS in self._goal_states:
            gs = self._goal_states[Goal.MINIMIZE_COLLISIONS]
            if collision_count > 3:
                gs.weight = min(2.0, gs.weight + self._adaptation_rate * 2)
                gs.reason = f"High collision count: {collision_count}"
            else:
                gs.weight = max(0.5, gs.weight - self._adaptation_rate)

        # ── Rule: Boost MINIMIZE_CONGESTION if zones are congested ────────────
        if Goal.MINIMIZE_CONGESTION in self._goal_states:
            gs = self._goal_states[Goal.MINIMIZE_CONGESTION]
            if len(congested) > 2:
                gs.weight = min(2.0, gs.weight + self._adaptation_rate)
                gs.reason = f"Congested zones: {congested[:3]}"
            else:
                gs.weight = max(0.3, gs.weight - self._adaptation_rate * 0.5)

        # ── Rule: Activate PRIORITIZE_EMERGENCY if emergency detected ─────────
        if emergency_detected:
            if Goal.PRIORITIZE_EMERGENCY not in self._goal_states:
                self._goal_states[Goal.PRIORITIZE_EMERGENCY] = GoalState(
                    goal=Goal.PRIORITIZE_EMERGENCY,
                    weight=1.5,
                    active=True,
                    reason="Emergency stop detected — activating priority goal",
                )
            else:
                self._goal_states[Goal.PRIORITIZE_EMERGENCY].weight = min(
                    2.0,
                    self._goal_states[Goal.PRIORITIZE_EMERGENCY].weight + 0.1
                )
        else:
            # Decay emergency goal when no emergency
            if Goal.PRIORITIZE_EMERGENCY in self._goal_states:
                gs = self._goal_states[Goal.PRIORITIZE_EMERGENCY]
                gs.weight = max(0.0, gs.weight - self._adaptation_rate)
                if gs.weight < 0.1:
                    del self._goal_states[Goal.PRIORITIZE_EMERGENCY]

        # ── Rule: Boost MAXIMIZE_THROUGHPUT when congestion is low ───────────
        if Goal.MAXIMIZE_THROUGHPUT in self._goal_states:
            gs = self._goal_states[Goal.MAXIMIZE_THROUGHPUT]
            if not congested and collision_count == 0:
                gs.weight = min(1.5, gs.weight + self._adaptation_rate * 0.5)
                gs.reason = "Low risk environment — prioritizing throughput"
            else:
                gs.weight = max(0.3, gs.weight - self._adaptation_rate)

    def get_action_biases(self) -> Dict[str, float]:
        """
        Compute aggregate action biases from all active goals.

        Each goal contributes its biases weighted by goal.weight.
        Final biases are normalized to [-0.3, +0.3] range.

        Returns:
            Dict of {action_name: bias_score}
            Positive bias → prefer this action
            Negative bias → avoid this action
        """
        aggregate: Dict[str, float] = {
            "EMERGENCY_STOP": 0.0,
            "REROUTE":        0.0,
            "HOLD":           0.0,
            "MONITOR":        0.0,
            "PRIORITIZE":     0.0,
        }

        total_weight = sum(
            gs.weight for gs in self._goal_states.values() if gs.active
        )
        if total_weight == 0:
            return aggregate

        for goal, gs in self._goal_states.items():
            if not gs.active or goal not in GOAL_BIASES:
                continue
            biases = GOAL_BIASES[goal]
            w = gs.weight / total_weight
            for action, bias in biases.items():
                aggregate[action] += w * bias

        # Clip to reasonable range
        return {
            action: float(np.clip(val, -0.30, 0.30))
            for action, val in aggregate.items()
        }

    def get_dominant_goal(self) -> Optional[Goal]:
        """Return the highest-weight active goal."""
        active = {g: gs for g, gs in self._goal_states.items() if gs.active}
        if not active:
            return None
        return max(active.keys(), key=lambda g: active[g].weight)

    def activate_goal(self, goal: Goal, weight: float = 1.0, reason: str = "") -> None:
        """Manually activate or update a goal."""
        self._goal_states[goal] = GoalState(
            goal=goal, weight=weight, active=True, reason=reason
        )
        logger.info(f"[GoalPlanner] Goal activated: {goal.value} (weight={weight})")

    def deactivate_goal(self, goal: Goal) -> None:
        """Deactivate a goal."""
        if goal in self._goal_states:
            self._goal_states[goal].active = False
            logger.info(f"[GoalPlanner] Goal deactivated: {goal.value}")

    def get_active_goals(self) -> List[dict]:
        """Return active goal states for API/dashboard."""
        return [
            gs.to_dict()
            for gs in self._goal_states.values()
            if gs.active
        ]

    def get_summary(self) -> dict:
        dominant = self.get_dominant_goal()
        return {
            "active_goal_count": sum(1 for gs in self._goal_states.values() if gs.active),
            "dominant_goal":     dominant.value if dominant else "NONE",
            "action_biases":     self.get_action_biases(),
            "active_goals":      self.get_active_goals(),
        }

    def update_goal_score(self, goal: Goal, score: float) -> None:
        """Update performance score for a goal (called by FeedbackEngine)."""
        if goal in self._goal_states:
            self._goal_states[goal].score = float(np.clip(score, 0.0, 1.0))
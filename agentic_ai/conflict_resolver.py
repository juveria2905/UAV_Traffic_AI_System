"""
================================================================================
agentic_ai/conflict_resolver.py — Agent Decision Conflict Resolution
================================================================================

PURPOSE:
  Resolves conflicts when multiple agents issue contradictory decisions
  for the same vehicle. Uses priority, confidence, and historical accuracy
  to determine the correct final decision.

EXAMPLE CONFLICT:
  MonitorAgent-A says EMERGENCY_STOP (confidence=0.91) for Track #7
  MonitorAgent-B says REROUTE        (confidence=0.88) for Track #7
  CoordinatorAgent needs to pick one.

RESOLUTION LOGIC:
  1. Always prefer EMERGENCY_STOP if ANY agent recommends it AND
     collision risk is genuinely HIGH (prevents false emergencies from
     overriding correct reroutes when risk is actually medium)
  2. Otherwise: weighted vote using priority × confidence × success_rate
  3. Tie-breaking: prefer more cautious action (REROUTE over HOLD, HOLD over MONITOR)

CONNECTS TO:
  agent_system.py (CoordinatorAgent) → calls resolve() per conflicted vehicle
  confidence_scorer.py → confidence is a resolution input
  memory/decision_memory.py → success_rate is a resolution input
================================================================================
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# RESOLUTION RESULT
# ==============================================================================

@dataclass
class ResolutionResult:
    """Result of a conflict resolution for one vehicle."""
    track_id:         int
    winning_action:   str
    winning_agent:    str
    final_confidence: float
    method_used:      str    # "emergency_override" | "weighted_vote" | "single_agent" | "priority_cascade"
    candidates:       int    # number of conflicting decisions
    reasoning:        str    # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "track_id":         self.track_id,
            "winning_action":   self.winning_action,
            "winning_agent":    self.winning_agent,
            "confidence":       round(self.final_confidence, 4),
            "method":           self.method_used,
            "candidates":       self.candidates,
            "reasoning":        self.reasoning,
        }


# ==============================================================================
# CONFLICT RESOLVER
# ==============================================================================

class ConflictResolver:
    """
    Resolves contradictory agent decisions for the same vehicle.

    PRIORITY CASCADE (from most to least cautious):
        EMERGENCY_STOP (5)
        REROUTE        (4)
        PRIORITIZE     (3)
        HOLD           (2)
        MONITOR        (1)

    When two agents disagree, the more cautious action wins IF
    the recommending agent has sufficient confidence AND the collision
    risk genuinely supports that level of caution.

    SAFETY BIAS:
        The resolver is biased toward caution. In traffic management,
        a false positive (unnecessary stop) is less costly than a false
        negative (missed collision). This mirrors aviation safety principles.
    """

    ACTION_PRIORITY = {
        "EMERGENCY_STOP": 5,
        "REROUTE":        4,
        "PRIORITIZE":     3,
        "HOLD":           2,
        "MONITOR":        1,
    }

    def __init__(
        self,
        emergency_min_confidence:  float = 0.35,
        min_confidence_to_act:     float = 0.50,
        safety_bias:               float = 0.10,   # adds to more-cautious actions
    ):
        """
        Args:
            emergency_min_confidence: Minimum confidence to accept EMERGENCY_STOP
            min_confidence_to_act:    Minimum confidence for non-emergency actions
            safety_bias:              Added to priority score of more-cautious decisions
        """
        self._emerg_min      = emergency_min_confidence
        self._min_conf       = min_confidence_to_act
        self._safety_bias    = safety_bias
        self._resolutions:   List[ResolutionResult] = []

        logger.info(
            f"ConflictResolver initialized | "
            f"emergency_min={emergency_min_confidence}, safety_bias={safety_bias}"
        )

    def resolve(
        self,
        track_id:          int,
        decisions:         List[dict],   # list of {action, confidence, agent, priority, success_rate}
        collision_risk:    float = 0.0,  # overall risk score [0,1]
    ) -> ResolutionResult:
        """
        Resolve conflicting decisions for one track.

        Args:
            track_id:       Vehicle track ID
            decisions:      List of candidate decisions (from multiple agents)
                            Each dict must have: action, confidence, agent, priority
                            Optional: success_rate (from memory)
            collision_risk: Overall collision risk for this track [0,1]

        Returns:
            ResolutionResult with the winning decision and explanation.
        """
        if not decisions:
            result = ResolutionResult(
                track_id=track_id,
                winning_action="MONITOR",
                winning_agent="ConflictResolver",
                final_confidence=1.0,
                method_used="no_candidates",
                candidates=0,
                reasoning="No candidates provided — defaulting to MONITOR",
            )
            return result

        if len(decisions) == 1:
            d = decisions[0]
            result = ResolutionResult(
                track_id=track_id,
                winning_action=d["action"],
                winning_agent=d.get("agent", "unknown"),
                final_confidence=d.get("confidence", 0.5),
                method_used="single_agent",
                candidates=1,
                reasoning=f"Single candidate: {d['action']} from {d.get('agent','?')}",
            )
            self._resolutions.append(result)
            return result

        # ── Rule 1: Emergency Override ────────────────────────────────────────
        # If any agent recommends EMERGENCY_STOP with sufficient confidence
        # AND the collision risk is genuinely high, always prefer it.
        emergency_candidates = [
            d for d in decisions
            if d["action"] == "EMERGENCY_STOP"
            and d.get("confidence", 0) >= self._emerg_min
        ]
        if emergency_candidates and collision_risk >= 0.60:
            best_emerg = max(emergency_candidates, key=lambda d: d.get("confidence", 0))
            result = ResolutionResult(
                track_id=track_id,
                winning_action="EMERGENCY_STOP",
                winning_agent=best_emerg.get("agent", "unknown"),
                final_confidence=best_emerg.get("confidence", 0.8),
                method_used="emergency_override",
                candidates=len(decisions),
                reasoning=(
                    f"Emergency override: {len(emergency_candidates)} agent(s) "
                    f"recommended EMERGENCY_STOP with collision_risk={collision_risk:.2f}"
                ),
            )
            self._resolutions.append(result)
            logger.warning(
                f"[Resolver] EMERGENCY OVERRIDE for Track #{track_id}: "
                f"{result.reasoning}"
            )
            return result

        # ── Rule 2: Priority Cascade with Safety Bias ─────────────────────────
        # Compute a weighted score for each candidate:
        # score = priority × confidence × success_rate + safety_bias
        scored = []
        for d in decisions:
            action      = d["action"]
            confidence  = d.get("confidence", 0.5)
            success_rate = d.get("success_rate", 0.5)
            priority    = self.ACTION_PRIORITY.get(action, 1)

            # Skip low-confidence non-emergency decisions
            if action != "EMERGENCY_STOP" and confidence < self._min_conf:
                continue

            # Safety bias: add bonus for more cautious actions
            # EMERGENCY(5) gets highest bonus, MONITOR(1) gets none
            bias_bonus = self._safety_bias * (priority / 5.0)

            score = float(priority) * confidence * success_rate + bias_bonus
            scored.append((score, d))

        if not scored:
            # All candidates filtered (low confidence) — default to MONITOR
            result = ResolutionResult(
                track_id=track_id,
                winning_action="MONITOR",
                winning_agent="ConflictResolver",
                final_confidence=0.5,
                method_used="fallback_monitor",
                candidates=len(decisions),
                reasoning="All candidates below confidence threshold — defaulting to MONITOR",
            )
            self._resolutions.append(result)
            return result

        # Pick highest-scoring candidate
        best_score, best_d = max(scored, key=lambda x: x[0])

        result = ResolutionResult(
            track_id=track_id,
            winning_action=best_d["action"],
            winning_agent=best_d.get("agent", "unknown"),
            final_confidence=best_d.get("confidence", 0.5),
            method_used="weighted_vote",
            candidates=len(decisions),
            reasoning=(
                f"Weighted vote from {len(scored)} candidates: "
                f"{best_d['action']} scored {best_score:.3f} "
                f"(conf={best_d.get('confidence',0):.2f}, "
                f"success={best_d.get('success_rate',0.5):.2f})"
            ),
        )
        self._resolutions.append(result)

        logger.debug(
            f"[Resolver] Track #{track_id}: {result.winning_action} wins "
            f"({result.method_used}, score={best_score:.3f})"
        )
        return result

    def resolve_batch(
        self,
        conflicts: Dict[int, List[dict]],
        risk_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[int, ResolutionResult]:
        """
        Resolve conflicts for multiple vehicles in one call.

        Args:
            conflicts:   {track_id: [decision_dict, ...]}
            risk_scores: {track_id: collision_risk_score}

        Returns:
            {track_id: ResolutionResult}
        """
        results = {}
        risk = risk_scores or {}
        for track_id, decisions in conflicts.items():
            results[track_id] = self.resolve(
                track_id=track_id,
                decisions=decisions,
                collision_risk=risk.get(track_id, 0.0),
            )
        return results

    def get_resolution_stats(self) -> dict:
        """Return statistics about resolution methods used."""
        if not self._resolutions:
            return {"total": 0}

        method_counts = defaultdict(int)
        action_counts = defaultdict(int)
        for r in self._resolutions:
            method_counts[r.method_used]      += 1
            action_counts[r.winning_action]   += 1

        return {
            "total_resolutions": len(self._resolutions),
            "method_counts":     dict(method_counts),
            "action_counts":     dict(action_counts),
        }
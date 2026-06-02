"""
================================================================================
agentic_ai/agent_system.py — Fully Integrated Hierarchical Agent System
================================================================================

INTEGRATION:
  This file wires together every agentic module into one runtime system.

  DECISION FLOW PER FRAME:
    tracks + collision_events + predictions
      ↓ PriorityEngine.compute_scores()
      ↓ ConfidenceScorer.score()   (per track)
      ↓ DecisionMemory.get_track_history()   (context)
      ↓ MonitorAgent.observe()  → local decisions
      ↓ ConflictResolver.resolve()   (per-vehicle)
      ↓ GoalPlanner.get_action_biases()   (bias adjustment)
      ↓ CoordinatorAgent.coordinate()
      ↓ ReasoningChain.build()  → logged
      ↓ ExecutorAgent.execute()
      ↓ DecisionMemory.store()
      ↓ FeedbackEngine.register_decision()
      ↓ MessageBus.flush()
      ↓ HierarchyManager.update()

  DECISION LOGIC (calibrated for realistic distribution):
    if TTC < 1.5s AND confidence > 0.80:  EMERGENCY_STOP
    elif priority_score > 0.70:           REROUTE
    elif density > 0.60:                  HOLD
    elif speed > 25px/frame:              HOLD
    else:                                 MONITOR
================================================================================
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from memory.decision_memory import DecisionMemory
from learning.feedback_engine import FeedbackEngine


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg, AGENT_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — every module may not exist on all deployments.
# We gracefully degrade if any advanced module is missing.
# ---------------------------------------------------------------------------

def _try_import(module_path: str, class_name: str):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Optional module not found: {module_path}.{class_name} — {e}")
        return None


# ==============================================================================
# ACTION TYPES
# ==============================================================================

class AgentAction(Enum):
    MONITOR        = "MONITOR"
    HOLD           = "HOLD"
    REROUTE        = "REROUTE"
    PRIORITIZE     = "PRIORITIZE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


ACTION_PRIORITY = {
    AgentAction.EMERGENCY_STOP: 5,
    AgentAction.REROUTE:        4,
    AgentAction.PRIORITIZE:     3,
    AgentAction.HOLD:           2,
    AgentAction.MONITOR:        1,
}


# ==============================================================================
# AGENT DECISION
# ==============================================================================

@dataclass
class AgentDecision:
    agent_name:  str
    track_id:    int
    action:      AgentAction
    reason:      str
    confidence:  float
    frame_idx:   int
    priority:    int   = 1
    timestamp:   float = field(default_factory=time.time)
    reasoning_chain_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "agent":        self.agent_name,
            "track_id":     self.track_id,
            "action":       self.action.value,
            "reason":       self.reason,
            "confidence":   round(self.confidence, 4),
            "frame_idx":    self.frame_idx,
            "priority":     self.priority,
            "timestamp":    round(self.timestamp, 3),
            "chain_id":     self.reasoning_chain_id,
        }


# ==============================================================================
# MONITOR AGENT
# ==============================================================================

class MonitorAgent:
    """
    Level 1 agent. Generates local decisions using calibrated thresholds.

    DECISION LOGIC (calibrated to produce realistic distribution):
      EMERGENCY_STOP → TTC < 1.5s AND confidence > 0.80
      REROUTE        → priority_score > 0.70 OR (TTC < 3.0s AND confidence > 0.55)
      HOLD           → density > 0.60 OR speed > 25px/frame
      MONITOR        → everything else
    """

    def __init__(self, agent_id: str, acfg=None):
        self.agent_id  = agent_id
        self.name      = f"MonitorAgent-{agent_id}"
        self.acfg      = acfg or cfg.agent
        self._history: List[AgentDecision] = []

    def observe(
        self,
        frame_idx:        int,
        tracks:           Dict[int, Any],
        collision_events: List[Any],
        predictions:      Dict[int, Any],
        priority_scores:  Dict[int, Any],     # {track_id: PriorityScore}
        confidence_scores: Dict[int, Any],    # {track_id: ConfidenceResult}
        memory_context:   Dict[int, Any],     # {track_id: track_summary_dict}
        goal_biases:      Dict[str, float],   # action → additive bias
    ) -> List[AgentDecision]:

        decisions: List[AgentDecision] = []
        acfg = self.acfg

        # Build TTC lookup per track from collision events
        track_ttcs: Dict[int, float]  = {}
        track_risks: Dict[int, str]   = {}
        for ev in collision_events:
            for tid in [ev.track_id_a, ev.track_id_b]:
                existing = track_ttcs.get(tid, float("inf"))
                if ev.ttc_seconds < existing:
                    track_ttcs[tid]  = ev.ttc_seconds
                    track_risks[tid] = ev.risk_level.value if hasattr(
                        ev.risk_level, "value") else str(ev.risk_level)

        for track_id, track in tracks.items():
            # ── Gather inputs ────────────────────────────────────────────────
            ttc       = track_ttcs.get(track_id, float("inf"))
            risk_str  = track_risks.get(track_id, "NONE")

            ps        = priority_scores.get(track_id)
            priority  = ps.final_score if ps else 0.0

            cs        = confidence_scores.get(track_id)
            confidence= cs.confidence if cs else 0.5

            vel       = track.velocity if hasattr(track, "velocity") else None
            speed     = float(np.sqrt(vel[0]**2 + vel[1]**2)) if vel else 0.0

            density   = ps.density_component if ps else 0.0

            mem       = memory_context.get(track_id, {})
            repeat    = mem.get("repeat_offender", False)
            consec    = mem.get("recent_actions", [])

            # ── Apply goal bias to confidence ─────────────────────────────────
            # goal biases shift the effective thresholds by adjusting confidence
            # positively for preferred actions
            bias_emergency = goal_biases.get("EMERGENCY_STOP", 0.0)
            bias_reroute   = goal_biases.get("REROUTE",        0.0)
            bias_hold      = goal_biases.get("HOLD",           0.0)

            # ── Decision logic (calibrated) ───────────────────────────────────
            action    = AgentAction.MONITOR
            reason    = "Nominal operation — no threats detected."
            conf_out  = confidence

            # Check consecutive reroute escalation
            # Check consecutive reroute escalation
            consecutive_reroutes = sum(
                1 for a in reversed(consec)
                if a == "REROUTE"
            )

            # Ignore ultra-small TTC duplicates
            if ttc <= 0.15:
                continue

            if (
                ttc < acfg.ttc_emergency_threshold
                and (confidence + bias_emergency)
                > acfg.emergency_confidence_min
            ) or (
                consecutive_reroutes >=
                acfg.consecutive_reroute_escalation
                and ttc < 3.0
                ):
                    action = AgentAction.EMERGENCY_STOP

                    reason = (
                        f"TTC={ttc:.2f}s below emergency threshold "
                        f"{acfg.ttc_emergency_threshold}s. "
                        f"conf={confidence:.2f}"
                    )

                    conf_out = min(
                        0.99,
                        confidence + 0.05
                    )

            elif (
                (priority + bias_reroute) > acfg.priority_reroute_score
                or (ttc < acfg.ttc_reroute_threshold and
                    (confidence + bias_reroute) > acfg.reroute_confidence_min)
                or repeat
            ):
                action   = AgentAction.REROUTE
                if repeat:
                    reason = (
                        f"Repeat offender pattern. priority={priority:.2f}, TTC={ttc:.2f}s"
                    )
                else:
                    reason = (
                        f"High priority ({priority:.2f}) or TTC={ttc:.2f}s < "
                        f"{acfg.ttc_reroute_threshold}s. conf={confidence:.2f}"
                    )
                conf_out = confidence

            elif (
                (density + bias_hold) > acfg.density_hold_threshold
                or speed > acfg.speed_hold_threshold
            ):
                action   = AgentAction.HOLD
                reason   = (
                    f"High density ({density:.2f}) or speed ({speed:.1f} px/frame). "
                    f"Requesting velocity hold."
                )
                conf_out = confidence

            dec = AgentDecision(
                agent_name=self.name,
                track_id=track_id,
                action=action,
                reason=reason,
                confidence=conf_out,
                frame_idx=frame_idx,
                priority=ACTION_PRIORITY[action],
            )
            decisions.append(dec)

        self._history.extend(decisions)
        return decisions


# ==============================================================================
# COORDINATOR AGENT
# ==============================================================================

class CoordinatorAgent:
    """
    Level 2 agent. Resolves conflicts, keeps highest-priority decision per vehicle.
    Applies goal biases and confidence filtering.
    """

    def __init__(self, acfg=None):
        self.name = "CoordinatorAgent"
        self.acfg = acfg or cfg.agent
        self._log: List[AgentDecision] = []

    def coordinate(
        self,
        frame_idx:       int,
        local_decisions: List[AgentDecision],
        tracks:          Dict[int, Any],
        resolver:        Optional[Any] = None,    # ConflictResolver
        goal_biases:     Dict[str, float] = None,
    ) -> List[AgentDecision]:

        biases = goal_biases or {}
        # Group by track_id
        per_track: Dict[int, List[AgentDecision]] = defaultdict(list)
        for dec in local_decisions:
            per_track[dec.track_id].append(dec)

        final: List[AgentDecision] = []

        for track_id, decs in per_track.items():
            decs.sort(key=lambda d: (d.priority, d.confidence), reverse=True)

            if resolver and len(decs) > 1:
                candidates = [
                    {
                        "action":       d.action.value,
                        "confidence":   d.confidence,
                        "agent":        d.agent_name,
                        "priority":     d.priority,
                        "success_rate": 0.5,
                    }
                    for d in decs
                ]
                result = resolver.resolve(track_id, candidates)
                best_action = AgentAction[result.winning_action]
                best_conf   = result.final_confidence
                best_reason = decs[0].reason
            else:
                best       = decs[0]
                best_action = best.action
                best_conf   = best.confidence
                best_reason = best.reason

            # Apply goal bias to confidence
            action_bias = biases.get(best_action.value, 0.0)
            adjusted_conf = float(np.clip(best_conf + action_bias * 0.5, 0.0, 1.0))

            # Filter low-confidence non-emergency decisions
            if (best_action != AgentAction.EMERGENCY_STOP
                    and adjusted_conf < self.acfg.hold_confidence_min):
                best_action = AgentAction.MONITOR
                best_reason = f"Low confidence ({adjusted_conf:.2f}) — downgraded to MONITOR"

            coordinated = AgentDecision(
                agent_name=self.name,
                track_id=track_id,
                action=best_action,
                reason=best_reason,
                confidence=adjusted_conf,
                frame_idx=frame_idx,
                priority=ACTION_PRIORITY[best_action],
            )
            final.append(coordinated)

        # Add MONITOR for tracks with no decisions
        for track_id in tracks:
            if track_id not in per_track:
                final.append(AgentDecision(
                    agent_name=self.name,
                    track_id=track_id,
                    action=AgentAction.MONITOR,
                    reason="No threats detected. Nominal operation.",
                    confidence=1.0,
                    frame_idx=frame_idx,
                    priority=1,
                ))

        final.sort(key=lambda d: d.priority, reverse=True)
        self._log.extend(final)
        return final


# ==============================================================================
# EXECUTOR AGENT
# ==============================================================================

class ExecutorAgent:
    """Level 3 agent. Finalises commands, logs, maintains state summary."""

    def __init__(self):
        self.name = "ExecutorAgent"
        self._executed: List[AgentDecision] = []
        self._latest: List[AgentDecision]   = []

    def execute(self, frame_idx: int, decisions: List[AgentDecision]) -> List[dict]:
        self._latest = decisions
        self._executed.extend(decisions)
        commands = []
        for dec in decisions:
            if dec.action in (AgentAction.EMERGENCY_STOP, AgentAction.REROUTE):
                logger.warning(
                    f"[EXECUTOR] {dec.action.value} → Track#{dec.track_id} "
                    f"| conf={dec.confidence:.2f} | {dec.reason}"
                )
            commands.append(dec.to_dict())
        return commands

    def get_system_state(self) -> dict:
        counts: Dict[str, int] = defaultdict(int)
        for d in self._latest:
            counts[d.action.value] += 1
        return {
            "frame":          self._latest[0].frame_idx if self._latest else 0,
            "total_vehicles": len(self._latest),
            "action_counts":  dict(counts),
            "emergency_stops": counts.get("EMERGENCY_STOP", 0),
            "reroutes":        counts.get("REROUTE", 0),
            "holds":           counts.get("HOLD", 0),
            "monitors":        counts.get("MONITOR", 0),
            "total_decisions": len(self._executed),
        }

    def save_decision_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        log = [d.to_dict() for d in self._executed]
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info(f"Saved {len(log)} decisions → {path}")

    @property
    def latest_decisions(self) -> List[AgentDecision]:
        return self._latest


# ==============================================================================
# HIERARCHICAL AGENT SYSTEM — MASTER ORCHESTRATOR
# ==============================================================================

class HierarchicalAgentSystem:
    """
    Orchestrates all agent levels plus all advanced modules:
      PriorityEngine, ConfidenceScorer, ReasoningChain, DecisionMemory,
      ConflictResolver, GoalPlanner, FeedbackEngine, HierarchyManager,
      MessageBus.

    Called once per frame from main.py.
    """

    def __init__(self, frame_width: int = 1280, frame_height: int = 720, acfg=None):
        self.acfg = acfg or cfg.agent
        self._frame_width  = frame_width
        self._frame_height = frame_height

        # ── Core agents ───────────────────────────────────────────────────────
        self.monitor     = MonitorAgent("Zone-Alpha", self.acfg)
        self.coordinator = CoordinatorAgent(self.acfg)
        self.executor    = ExecutorAgent()

        # ── Advanced modules (graceful degradation if missing) ────────────────
        # ===============================
        # Core communication first
        # ===============================

        self._message_bus = self._init_message_bus()

        # ===============================
        # Decision engines
        # ===============================

        self._priority_engine = self._init_priority_engine()

        self._confidence_scorer = self._init_confidence_scorer()

        self._conflict_resolver = self._init_conflict_resolver()

        self._goal_planner = self._init_goal_planner()

        # ===============================
        # Memory / learning
        # ===============================

        self._decision_memory = self._init_decision_memory()

        self._feedback_engine = self._init_feedback_engine()

        # ===============================
        # Reasoning
        # ===============================

        self._reasoning_reg = self._init_reasoning_registry()

        # ===============================
        # Hierarchy LAST
        # ===============================

        self._hierarchy_manager = self._init_hierarchy_manager()

        self._frame_count = 0
        logger.info(
            f"HierarchicalAgentSystem ready | "
            f"frame={frame_width}×{frame_height} | "
            f"modules: priority={'✓' if self._priority_engine else '✗'} "
            f"confidence={'✓' if self._confidence_scorer else '✗'} "
            f"memory={'✓' if self._decision_memory else '✗'} "
            f"reasoning={'✓' if self._reasoning_reg else '✗'} "
            f"feedback={'✓' if self._feedback_engine else '✗'} "
            f"hierarchy={'✓' if self._hierarchy_manager else '✗'}"
        )

    # ── Module initialization helpers ──────────────────────────────────────────

    def _init_priority_engine(self):
        if not self.acfg.enable_confidence_scorer:
            return None
        cls = _try_import("agentic_ai.priority_engine", "PriorityEngine")
        return cls() if cls else None

    def _init_confidence_scorer(self):
        if not self.acfg.enable_confidence_scorer:
            return None
        cls = _try_import("agentic_ai.confidence_scorer", "ConfidenceScorer")
        return cls() if cls else None

    def _init_reasoning_registry(self):
        if not self.acfg.enable_reasoning_chains:
            return None
        fn = _try_import("agentic_ai.reasoning_chain", "get_reasoning_registry")
        return fn() if fn else None

    def _init_decision_memory(self):
        if not self.acfg.enable_decision_memory:
            return None

        try:
            return DecisionMemory()
        except Exception as e:
            logger.warning(f"DecisionMemory init failed: {e}")
            return None

    def _init_conflict_resolver(self):
        if not self.acfg.enable_conflict_resolver:
            return None
        cls = _try_import("agentic_ai.conflict_resolver", "ConflictResolver")
        return cls() if cls else None

    def _init_goal_planner(self):
        if not self.acfg.enable_goal_planner:
            return None
        cls = _try_import("agentic_ai.goal_planner", "GoalPlanner")
        return cls() if cls else None

    def _init_feedback_engine(self):
        if not self.acfg.enable_feedback_engine:
            return None

        try:
            return FeedbackEngine()
        except Exception as e:
            logger.warning(f"FeedbackEngine init failed: {e}")
            return None

    def _init_hierarchy_manager(self):
        if not self.acfg.enable_hierarchy_manager:
            return None
        cls = _try_import("agentic_ai.hierarchy_manager", "HierarchyManager")
        if cls:
            bus = self._message_bus
            return cls(
                frame_width=self._frame_width,
                frame_height=self._frame_height,
                bus=bus,
                grid_cols=self.acfg.grid_cols,
                grid_rows=self.acfg.grid_rows,
            )
        return None

    def _init_message_bus(self):
        if not self.acfg.enable_message_bus:
            return None
        fn = _try_import("agentic_ai.communication", "get_message_bus")
        return fn() if fn else None

    # ── MAIN PER-FRAME ENTRY POINT ─────────────────────────────────────────────

    def process_frame(
        self,
        frame_idx:        int,
        tracks:           Dict[int, Any],
        collision_events: List[Any],
        predictions:      Dict[int, Any],
    ) -> dict:
        """
        Full agentic pipeline for one frame.

        Returns dict with:
          decisions:        list of command dicts
          system_state:     counts/summary
          latest_decisions: AgentDecision objects (for annotation)
          reasoning_chains: recent chain dicts
          city_state:       hierarchy zone state
          goal_summary:     active goals + biases
          memory_stats:     memory system stats
          feedback_stats:   feedback learning stats
        """
        self._frame_count += 1
        t_start = time.perf_counter()

        # ── Step 1: Priority scores ────────────────────────────────────────────
        priority_scores: Dict[int, Any] = {}
        if self._priority_engine and tracks:
            try:
                priority_scores = self._priority_engine.compute_scores(
                    frame_idx=frame_idx,
                    tracks=tracks,
                    collision_events=collision_events,
                )
            except Exception as e:
                logger.error(f"PriorityEngine error: {e}")

        # ── Step 2: Confidence scores ──────────────────────────────────────────
        confidence_scores: Dict[int, Any] = {}
        if self._confidence_scorer and tracks:
            # Build TTC per track
            track_ttcs = self._build_track_ttcs(collision_events)
            for track_id, track in tracks.items():
                try:
                    ps  = priority_scores.get(track_id)
                    vel = track.velocity if hasattr(track, "velocity") else None
                    spd = float(np.sqrt(vel[0]**2 + vel[1]**2)) if vel else 0.0
                    pred = predictions.get(track_id)
                    pred_conf = pred.confidence if pred and hasattr(pred, "confidence") else 0.5

                    cs = self._confidence_scorer.score(
                        track_id=track_id,
                        action="REROUTE",    # score generically; action fine-tuned later
                        ttc_seconds=track_ttcs.get(track_id),
                        ttc_threshold=cfg.collision.ttc_high_risk_seconds,
                        speed=spd,
                        speed_variance=ps.speed_variance_component if ps else 0.0,
                        density=ps.density_component if ps else 0.0,
                        historical_accuracy=(
                            self._decision_memory.get_success_rate(track_id)
                            if self._decision_memory else 0.5
                        ),
                        prediction_confidence=pred_conf,
                        collision_risk_score=ps.risk_component if ps else 0.0,
                    )
                    confidence_scores[track_id] = cs
                except Exception as e:
                    logger.debug(f"ConfidenceScorer error track {track_id}: {e}")

        # ── Step 3: Memory context ─────────────────────────────────────────────
        memory_context: Dict[int, dict] = {}
        if self._decision_memory:
            for track_id in tracks:
                summary = self._decision_memory.get_track_summary(track_id)
                if summary:
                    memory_context[track_id] = summary

        # ── Step 4: Goal planner ───────────────────────────────────────────────
        goal_biases: Dict[str, float] = {}
        goal_summary: dict = {}
        if self._goal_planner:
            try:
                congested = (
                    self._hierarchy_manager.get_congested_zones()
                    if self._hierarchy_manager else []
                )
                emergency_active = any(
                    ev.risk_level.value == "HIGH"
                    if hasattr(ev.risk_level, "value") else ev.risk_level == "HIGH"
                    for ev in collision_events
                )
                self._goal_planner.update(
                    collision_count=len(collision_events),
                    congestion_zones=congested,
                    emergency_detected=emergency_active,
                    frame_idx=frame_idx,
                )
                goal_biases = self._goal_planner.get_action_biases()
                goal_summary = self._goal_planner.get_summary()
            except Exception as e:
                logger.debug(f"GoalPlanner error: {e}")

        # ── Step 5: Monitor (local decisions) ─────────────────────────────────
        local_decisions = self.monitor.observe(
            frame_idx=frame_idx,
            tracks=tracks,
            collision_events=collision_events,
            predictions=predictions,
            priority_scores=priority_scores,
            confidence_scores=confidence_scores,
            memory_context=memory_context,
            goal_biases=goal_biases,
        )

        # ── Step 6: Coordinate (conflict resolution + goal biases) ─────────────
        final_decisions = self.coordinator.coordinate(
            frame_idx=frame_idx,
            local_decisions=local_decisions,
            tracks=tracks,
            resolver=self._conflict_resolver,
            goal_biases=goal_biases,
        )

        # ── Step 7: Build reasoning chains ────────────────────────────────────
        if self._reasoning_reg and cfg.agent.enable_reasoning_chains:
            self._build_reasoning_chains(
                frame_idx=frame_idx,
                decisions=final_decisions,
                priority_scores=priority_scores,
                confidence_scores=confidence_scores,
                collision_events=collision_events,
            )

        # ── Step 8: Execute ────────────────────────────────────────────────────
        commands = self.executor.execute(frame_idx, final_decisions)
        system_state = self.executor.get_system_state()

        # ── Step 9: Store in memory ────────────────────────────────────────────
        if self._decision_memory:
            self._store_decisions_in_memory(
                frame_idx=frame_idx,
                decisions=final_decisions,
                priority_scores=priority_scores,
                confidence_scores=confidence_scores,
                collision_events=collision_events,
            )

        # ── Step 10: Register for feedback ───────────────────────────────────
        track_ttcs_for_feedback = self._build_track_ttcs(collision_events)
        if self._feedback_engine:
            for dec in final_decisions:
                if dec.action != AgentAction.MONITOR:
                    try:
                        ps   = priority_scores.get(dec.track_id)
                        self._feedback_engine.register_decision(
                            track_id=dec.track_id,
                            frame_idx=frame_idx,
                            action=dec.action.value,
                            ttc_at_time=track_ttcs_for_feedback.get(dec.track_id, 99.0),
                            density=ps.density_component if ps else 0.0,
                            confidence=dec.confidence,
                        )
                    except Exception as e:
                        logger.debug(f"FeedbackEngine register error: {e}")

            # Evaluate pending feedback
            try:
                feedback_records = self._feedback_engine.evaluate(
                    current_frame=frame_idx,
                    current_ttcs=track_ttcs_for_feedback,
                )
                # Write outcomes back to memory
                if feedback_records and self._decision_memory:
                    for rec in feedback_records:
                        self._decision_memory.update_outcome(
                            rec.track_id, rec.frame_idx, rec.reward
                        )
            except Exception as e:
                logger.debug(f"FeedbackEngine evaluate error: {e}")

        # ── Step 11: Hierarchy manager ─────────────────────────────────────────
        city_state: dict = {}
        if self._hierarchy_manager:
            try:
                city_state = self._hierarchy_manager.update(
                    frame_idx=frame_idx,
                    tracks=tracks,
                    collision_events=collision_events,
                )
            except Exception as e:
                logger.debug(f"HierarchyManager error: {e}")

        # ── Step 12: Message bus flush ─────────────────────────────────────────
        if self._message_bus:
            try:
                self._message_bus.flush()
            except Exception as e:
                logger.debug(f"MessageBus flush error: {e}")

        # ── Timing log ────────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if self._frame_count % 100 == 0:
            logger.info(
                f"[AgentSystem] Frame {frame_idx}: "
                f"{len(final_decisions)} decisions in {elapsed_ms:.1f}ms | "
                f"state={system_state}"
            )

        return {
            "decisions":        commands,
            "system_state":     system_state,
            "latest_decisions": self.executor.latest_decisions,
            "reasoning_chains": (
                self._reasoning_reg.get_recent(max_count=10)
                if self._reasoning_reg else []
            ),
            "city_state":       city_state,
            "goal_summary":     goal_summary,
            "memory_stats":     (
                self._decision_memory.get_global_stats()
                if self._decision_memory else {}
            ),
            "feedback_stats":   (
                self._feedback_engine.get_stats()
                if self._feedback_engine else {}
            ),
            "agent_latency_ms": round(elapsed_ms, 2),
        }

    # ── Frame annotation ───────────────────────────────────────────────────────

    def annotate_frame(self, frame, agent_output: dict, tracks: Dict[int, Any]):
        """Draw agent decisions and hierarchy grid on frame."""
        import cv2
        import numpy as np

        annotated = frame.copy()
        decisions = agent_output.get("latest_decisions", [])

        action_colors = {
            AgentAction.EMERGENCY_STOP: (0,   0, 255),
            AgentAction.REROUTE:        (0, 165, 255),
            AgentAction.HOLD:           (0, 255, 255),
            AgentAction.PRIORITIZE:     (255, 0, 255),
            AgentAction.MONITOR:        (0, 255, 0),
        }

        for dec in decisions:
            if dec.action == AgentAction.MONITOR:
                continue

            track = tracks.get(dec.track_id)
            if track is None or not hasattr(track, "bbox_xyxy") or track.bbox_xyxy is None:
                continue

            x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
            color = action_colors.get(dec.action, (255, 255, 255))
            thick = 3 if dec.action == AgentAction.EMERGENCY_STOP else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

            label = f"[{dec.action.value}] {dec.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA,
            )

        # Hierarchy zone overlay
        if self._hierarchy_manager:
            try:
                annotated = self._hierarchy_manager.annotate_frame(annotated, alpha=0.10)
            except Exception:
                pass

        # Bottom HUD
        state = agent_output.get("system_state", {})
        h, w  = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        hud = (
            f"AGENTS | "
            f"EMERGENCY: {state.get('emergency_stops', 0)} | "
            f"REROUTE: {state.get('reroutes', 0)} | "
            f"HOLD: {state.get('holds', 0)} | "
            f"MONITOR: {state.get('monitors', 0)}"
        )
        cv2.putText(
            annotated, hud, (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 1, cv2.LINE_AA,
        )
        return annotated

    def save_logs(self, output_dir: Path) -> None:
        """Save all decision logs and memory exports."""
        output_dir = Path(output_dir)
        self.executor.save_decision_log(output_dir / cfg.agent.decisions_json)

        if self._decision_memory:
            try:
                self._decision_memory.export_to_json(cfg.paths.memory_export_json)
            except Exception as e:
                logger.warning(f"Memory export failed: {e}")

        if self._reasoning_reg:
            try:
                self._reasoning_reg.export_json(cfg.paths.reasoning_json)
            except Exception as e:
                logger.warning(f"Reasoning export failed: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_track_ttcs(self, collision_events: List[Any]) -> Dict[int, float]:
        """Build {track_id: min_ttc} lookup from collision events."""
        ttcs: Dict[int, float] = {}
        for ev in collision_events:
            ttc = getattr(ev, "ttc_seconds", 99.0)
            for tid in [ev.track_id_a, ev.track_id_b]:
                if tid not in ttcs or ttc < ttcs[tid]:
                    ttcs[tid] = ttc
        return ttcs

    def _build_reasoning_chains(
        self,
        frame_idx:        int,
        decisions:        List[AgentDecision],
        priority_scores:  dict,
        confidence_scores: dict,
        collision_events: list,
    ) -> None:
        """Build and register reasoning chains for non-MONITOR decisions."""
        ChainBuilder = _try_import("agentic_ai.reasoning_chain", "ChainBuilder")
        if ChainBuilder is None:
            return

        track_ttcs = self._build_track_ttcs(collision_events)

        for dec in decisions:
            if dec.action == AgentAction.MONITOR:
                continue
            try:
                ps  = priority_scores.get(dec.track_id)
                cs  = confidence_scores.get(dec.track_id)
                ttc = track_ttcs.get(dec.track_id)

                chain = (
                    ChainBuilder(
                        track_id=dec.track_id,
                        frame_idx=frame_idx,
                        agent=dec.agent_name,
                    )
                    .observe("track_id", dec.track_id, "")
                )

                if ps:
                    chain.analyze_priority(
                        ps.final_score,
                        {"risk": ps.risk_component, "speed_var": ps.speed_variance_component,
                         "density": ps.density_component, "history": ps.historical_component}
                    )

                if ttc is not None:
                    chain.analyze_ttc(ttc, cfg.collision.ttc_high_risk_seconds)

                if cs:
                    chain.add_context(
                        f"Confidence breakdown: {cs.explain()}",
                        metadata=cs.to_dict()
                    )

                chain.decide(dec.action.value, dec.confidence)
                built = chain.build()
                dec.reasoning_chain_id = built.chain_id
                self._reasoning_reg.register(built)

            except Exception as e:
                logger.debug(f"Reasoning chain build error track {dec.track_id}: {e}")

    def _store_decisions_in_memory(
        self,
        frame_idx: int,
        decisions: List[AgentDecision],
        priority_scores: dict,
        confidence_scores: dict,
        collision_events: list,
    ) -> None:

        if self._decision_memory is None:
            return

        track_ttcs = self._build_track_ttcs(collision_events)

        for dec in decisions:
            try:
                ps = priority_scores.get(dec.track_id)

                record = {
                    "track_id": dec.track_id,
                    "frame_idx": frame_idx,
                    "action": dec.action.value,
                    "confidence": dec.confidence,
                    "agent_name": dec.agent_name,
                    "ttc": track_ttcs.get(dec.track_id, 99.0),
                    "priority_score": (
                        ps.final_score if ps else 0.0
                    ),
                    "density": (
                        ps.density_component if ps else 0.0
                    )
                }

                self._decision_memory.store(record)

            except Exception as e:
                logger.debug(
                    f"Memory store error track {dec.track_id}: {e}"
                )

    # ── Accessors for API ─────────────────────────────────────────────────────

    def get_reasoning_history(self, max_count: int = 20) -> List[dict]:
        if self._reasoning_reg:
            return self._reasoning_reg.get_recent(max_count=max_count)
        return []

    def get_memory_stats(self) -> dict:
        if self._decision_memory:
            return self._decision_memory.get_global_stats()
        return {}

    def get_memory_summaries(self) -> List[dict]:
        if self._decision_memory:
            return self._decision_memory.get_all_summaries()
        return []

    def get_confidence_weights(self) -> dict:
        if self._confidence_scorer:
            return self._confidence_scorer.get_weights()
        return {}

    def get_feedback_stats(self) -> dict:
        if self._feedback_engine:
            return self._feedback_engine.get_stats()
        return {}

    def get_feedback_recent(self, n: int = 20) -> List[dict]:
        if self._feedback_engine:
            return self._feedback_engine.get_recent_feedback(n)
        return []

    def get_goal_summary(self) -> dict:
        if self._goal_planner:
            return self._goal_planner.get_summary()
        return {}

    def get_priority_weights(self) -> dict:
        if self._priority_engine:
            return self._priority_engine.get_current_weights()
        return {}

    def get_message_bus_stats(self) -> dict:
        if self._message_bus:
            return self._message_bus.get_stats()
        return {}

    def get_city_state(self) -> dict:
        if self._hierarchy_manager:
            return self._hierarchy_manager.latest_city_state
        return {}

    def get_hierarchy_stats(self) -> dict:
        if self._hierarchy_manager:
            return self._hierarchy_manager.get_stats()
        return {}
"""
================================================================================
agentic_ai/reasoning_chain.py — Explainable AI Reasoning Chain System
================================================================================

PURPOSE:
  Records the step-by-step reasoning behind every agent decision.
  Makes the AI system explainable — critical for research presentations,
  internship interviews, and SIH evaluation.

WHY THIS FILE EXISTS:
  Traditional AI systems are black boxes: input → output.
  A reasoning chain makes the decision process transparent:
    "Track #7 was given REROUTE because:
     Step 1: TTC = 1.4s → HIGH RISK
     Step 2: Relative speed = 4.1 px/frame → CONVERGING
     Step 3: Density = 0.8 → CONGESTED ZONE
     Step 4: Priority score = 0.91 → CRITICAL
     Step 5: Decision = REROUTE (EMERGENCY_STOP saved for TTC < 1.0s)
     Confidence: 0.94"

WHAT BREAKS IF REMOVED:
  - No explainability for agent decisions
  - Cannot trace why a vehicle was flagged
  - No audit trail for decision review
  - Dashboard reasoning tab becomes empty

CONNECTS TO:
  - agent_system.py     → MonitorAgent, CoordinatorAgent log reasoning steps
  - priority_engine.py  → Priority scores added as reasoning steps
  - api.py             → GET /api/reasoning returns chain history
  - dashboard/app.py   → "Reasoning Chains" tab displays chains
================================================================================
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# STEP TYPES
# ==============================================================================

class StepType(Enum):
    """
    Classification of reasoning step types.
    Used for color-coding in dashboard visualization.
    """
    OBSERVATION   = "OBSERVATION"   # Raw sensor/detection data
    ANALYSIS      = "ANALYSIS"      # Computation result
    COMPARISON    = "COMPARISON"    # Threshold comparison
    DECISION      = "DECISION"      # Final decision made
    ESCALATION    = "ESCALATION"    # Passed to higher agent level
    CONFIDENCE    = "CONFIDENCE"    # Confidence assessment
    CONTEXT       = "CONTEXT"       # Contextual information


# ==============================================================================
# REASONING STEP
# ==============================================================================

@dataclass
class ReasoningStep:
    """
    A single step in the reasoning chain.

    EXAMPLE:
        ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="TTC computed",
            value=1.4,
            unit="seconds",
            interpretation="HIGH RISK: TTC below 3.0s threshold",
            confidence=0.92,
        )

    Fields:
        step_num:       Sequential step number within this chain
        step_type:      Category of step (observation, analysis, decision, etc.)
        description:    What this step computes or observes
        value:          Numeric result (if applicable)
        unit:           Unit of the value ("seconds", "px/frame", "dimensionless")
        interpretation: Human-readable meaning of the value
        confidence:     How reliable this step's output is [0, 1]
        timestamp:      When this step was recorded
        metadata:       Any additional context (dict)
    """
    step_num:       int
    step_type:      StepType
    description:    str
    value:          Optional[float]  = None
    unit:           Optional[str]    = None
    interpretation: Optional[str]   = None
    confidence:     float            = 1.0
    timestamp:      float            = field(default_factory=time.time)
    metadata:       Dict[str, Any]   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step":           self.step_num,
            "type":           self.step_type.value,
            "description":    self.description,
            "value":          round(self.value, 4) if self.value is not None else None,
            "unit":           self.unit,
            "interpretation": self.interpretation,
            "confidence":     round(self.confidence, 4),
            "timestamp":      round(self.timestamp, 4),
            "metadata":       self.metadata,
        }

    def __str__(self) -> str:
        val_str = f" = {self.value:.3f} {self.unit}" if self.value is not None else ""
        interp  = f" → {self.interpretation}" if self.interpretation else ""
        conf    = f" [conf={self.confidence:.2f}]"
        return f"  Step {self.step_num} [{self.step_type.value}] {self.description}{val_str}{interp}{conf}"


# ==============================================================================
# REASONING CHAIN
# ==============================================================================

@dataclass
class ReasoningChain:
    """
    Complete reasoning chain for one agent decision on one track.

    This is the core "explainability artifact" of the system.

    EXAMPLE OUTPUT:
        ReasoningChain:
          Track #7 | Frame 221 | MonitorAgent-Zone-Alpha
          Step 1 [OBSERVATION]  Current position = (420, 315) px
          Step 2 [OBSERVATION]  Velocity = (4.1, 2.3) px/frame
          Step 3 [ANALYSIS]     Speed = 4.71 px/frame → MODERATE
          Step 4 [ANALYSIS]     TTC = 1.40 seconds → HIGH RISK
          Step 5 [ANALYSIS]     Priority score = 0.91 → CRITICAL
          Step 6 [COMPARISON]   TTC < 3.0s threshold → ESCALATE
          Step 7 [DECISION]     Action = REROUTE (confidence=0.94)
          Step 8 [ESCALATION]   Forwarded to CoordinatorAgent
    """
    chain_id:       str                = field(default_factory=lambda: str(uuid.uuid4())[:8])
    track_id:       int                = 0
    frame_idx:      int                = 0
    agent_name:     str                = ""
    final_decision: Optional[str]      = None
    final_confidence: float            = 0.0
    steps:          List[ReasoningStep] = field(default_factory=list)
    created_at:     float              = field(default_factory=time.time)
    completed:      bool               = False

    def add_step(
        self,
        description:    str,
        step_type:      StepType        = StepType.ANALYSIS,
        value:          Optional[float] = None,
        unit:           Optional[str]   = None,
        interpretation: Optional[str]   = None,
        confidence:     float           = 1.0,
        metadata:       Optional[dict]  = None,
    ) -> "ReasoningChain":
        """
        Add a reasoning step.

        Returns self for method chaining:
            chain.add_step(...).add_step(...).conclude(...)

        Args:
            description:    What this step does
            step_type:      Type of step
            value:          Numeric result
            unit:           Unit for the value
            interpretation: Human-readable meaning
            confidence:     Reliability of this step's output
            metadata:       Extra context dict
        """
        step = ReasoningStep(
            step_num=len(self.steps) + 1,
            step_type=step_type,
            description=description,
            value=value,
            unit=unit,
            interpretation=interpretation,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return self

    def conclude(self, decision: str, confidence: float) -> "ReasoningChain":
        """
        Mark the chain as complete with a final decision.
        Automatically adds a DECISION step.
        """
        self.final_decision  = decision
        self.final_confidence = confidence
        self.completed        = True
        self.add_step(
            description=f"Final Decision: {decision}",
            step_type=StepType.DECISION,
            confidence=confidence,
            interpretation=f"Confidence = {confidence:.2%}",
        )
        return self

    @property
    def duration_ms(self) -> float:
        """Time taken to build this chain in milliseconds."""
        if self.steps:
            return (self.steps[-1].timestamp - self.created_at) * 1000
        return 0.0

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all steps."""
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)

    def to_dict(self) -> dict:
        return {
            "chain_id":        self.chain_id,
            "track_id":        self.track_id,
            "frame_idx":       self.frame_idx,
            "agent":           self.agent_name,
            "decision":        self.final_decision,
            "confidence":      round(self.final_confidence, 4),
            "avg_confidence":  round(self.avg_confidence, 4),
            "steps":           [s.to_dict() for s in self.steps],
            "step_count":      len(self.steps),
            "duration_ms":     round(self.duration_ms, 2),
            "completed":       self.completed,
            "created_at":      round(self.created_at, 3),
        }

    def to_text(self) -> str:
        """Human-readable text format for logs and debug output."""
        lines = [
            f"\n{'═' * 60}",
            f"REASONING CHAIN [{self.chain_id}]",
            f"Track #{self.track_id} | Frame {self.frame_idx} | Agent: {self.agent_name}",
            f"{'─' * 60}",
        ]
        for step in self.steps:
            lines.append(str(step))
        lines += [
            f"{'─' * 60}",
            f"FINAL: {self.final_decision} (confidence={self.final_confidence:.2%})",
            f"{'═' * 60}\n",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ==============================================================================
# REASONING CHAIN BUILDER (FLUENT API)
# ==============================================================================

class ChainBuilder:
    """
    Fluent builder for constructing reasoning chains.
    Makes chain construction readable and concise.

    USAGE IN AGENT:
        chain = (
            ChainBuilder(track_id=7, frame_idx=221, agent="MonitorAgent-Alpha")
            .observe("position", (420, 315), "px")
            .observe_velocity(4.1, 2.3)
            .analyze_ttc(1.4, threshold=3.0)
            .analyze_priority(0.91)
            .compare_threshold(1.4, 3.0, "TTC", "HIGH")
            .decide("REROUTE", confidence=0.94)
            .build()
        )
    """

    def __init__(self, track_id: int, frame_idx: int, agent: str):
        self._chain = ReasoningChain(
            track_id=track_id,
            frame_idx=frame_idx,
            agent_name=agent,
        )

    def observe(
        self, what: str, value: Any, unit: str = "",
        confidence: float = 1.0,
    ) -> "ChainBuilder":
        """Record an observation of sensor/detection data."""
        self._chain.add_step(
            description=f"Observed {what}: {value}",
            step_type=StepType.OBSERVATION,
            value=float(value) if isinstance(value, (int, float)) else None,
            unit=unit,
            confidence=confidence,
        )
        return self

    def observe_velocity(self, vx: float, vy: float) -> "ChainBuilder":
        """Record velocity observation."""
        import math
        speed = math.sqrt(vx**2 + vy**2)
        self._chain.add_step(
            description=f"Velocity observed: vx={vx:.2f}, vy={vy:.2f}",
            step_type=StepType.OBSERVATION,
            value=speed,
            unit="px/frame",
            interpretation=f"Speed = {speed:.2f} px/frame | "
                           f"Angle = {math.degrees(math.atan2(vy, vx)):.1f}°",
        )
        return self

    def analyze_ttc(self, ttc: float, threshold_high: float = 3.0,
                    threshold_medium: float = 6.0) -> "ChainBuilder":
        """Record TTC analysis step."""
        if ttc <= threshold_high:
            interp = f"HIGH RISK — TTC below {threshold_high}s emergency threshold"
            conf   = min(0.99, 1.0 - ttc / (threshold_high * 2))
        elif ttc <= threshold_medium:
            interp = f"MEDIUM RISK — TTC between {threshold_high}s and {threshold_medium}s"
            conf   = 0.75
        else:
            interp = f"LOW RISK — TTC above {threshold_medium}s safety margin"
            conf   = 0.90

        self._chain.add_step(
            description="Time-to-Collision (TTC) computed",
            step_type=StepType.ANALYSIS,
            value=ttc,
            unit="seconds",
            interpretation=interp,
            confidence=conf,
        )
        return self

    def analyze_priority(self, score: float, components: Optional[dict] = None) -> "ChainBuilder":
        """Record priority score analysis."""
        if score >= 0.8:     level = "CRITICAL"
        elif score >= 0.6:   level = "HIGH"
        elif score >= 0.4:   level = "MEDIUM"
        else:                level = "LOW/NOMINAL"

        self._chain.add_step(
            description="Multi-factor priority score computed",
            step_type=StepType.ANALYSIS,
            value=score,
            unit="dimensionless",
            interpretation=f"Priority level: {level}",
            confidence=0.90,
            metadata=components or {},
        )
        return self

    def analyze_density(self, density: float, nearby_count: int) -> "ChainBuilder":
        """Record traffic density step."""
        self._chain.add_step(
            description=f"Local traffic density: {nearby_count} vehicles nearby",
            step_type=StepType.ANALYSIS,
            value=density,
            unit="normalized [0,1]",
            interpretation="CONGESTED" if density > 0.6 else (
                "MODERATE" if density > 0.3 else "SPARSE"
            ),
        )
        return self

    def compare_threshold(
        self,
        value:     float,
        threshold: float,
        label:     str,
        result:    str,
    ) -> "ChainBuilder":
        """Record a threshold comparison decision step."""
        direction = "below" if value < threshold else "above"
        self._chain.add_step(
            description=f"{label} {direction} threshold ({threshold})",
            step_type=StepType.COMPARISON,
            value=value,
            interpretation=f"→ {result}",
            confidence=0.95,
            metadata={"threshold": threshold, "actual": value, "result": result},
        )
        return self

    def add_context(self, context: str, metadata: Optional[dict] = None) -> "ChainBuilder":
        """Add contextual information."""
        self._chain.add_step(
            description=context,
            step_type=StepType.CONTEXT,
            metadata=metadata or {},
        )
        return self

    def escalate(self, to_agent: str) -> "ChainBuilder":
        """Record escalation to a higher agent level."""
        self._chain.add_step(
            description=f"Decision escalated to {to_agent}",
            step_type=StepType.ESCALATION,
            interpretation=f"Local decision forwarded up the hierarchy to {to_agent}",
        )
        return self

    def decide(self, action: str, confidence: float) -> "ChainBuilder":
        """Record final decision and mark chain complete."""
        self._chain.conclude(action, confidence)
        return self

    def build(self) -> ReasoningChain:
        """Return the completed ReasoningChain object."""
        if not self._chain.completed:
            logger.warning(
                f"[ReasoningChain] Chain for Track #{self._chain.track_id} "
                f"built without calling .decide(). Marking as incomplete."
            )
        return self._chain


# ==============================================================================
# REASONING CHAIN REGISTRY (THREAD-SAFE STORE)
# ==============================================================================

class ReasoningChainRegistry:
    """
    Stores all reasoning chains for retrieval by dashboard and API.

    Maintains:
      - recent_chains:  Latest N chains (rolling buffer, fast lookup)
      - chain_by_track: Most recent chain per track_id (for dashboard table)
      - statistics:     Decision type counts, average confidence, etc.
    """

    def __init__(self, maxlen: int = 200):
        import threading
        self._lock       = threading.Lock()
        self._recent:    deque[ReasoningChain]     = deque(maxlen=maxlen)
        self._by_track:  Dict[int, ReasoningChain] = {}
        self._decision_counts: Dict[str, int]      = {}
        self._total = 0

    def register(self, chain: ReasoningChain) -> None:
        """Store a completed reasoning chain."""
        with self._lock:
            self._recent.append(chain)
            self._by_track[chain.track_id] = chain
            decision = chain.final_decision or "UNKNOWN"
            self._decision_counts[decision] = self._decision_counts.get(decision, 0) + 1
            self._total += 1

        logger.debug(
            f"[ReasoningRegistry] Chain {chain.chain_id} registered: "
            f"Track #{chain.track_id} → {chain.final_decision} "
            f"(conf={chain.final_confidence:.2f})"
        )

    def get_recent(
        self,
        max_count: int = 20,
        track_id: Optional[int] = None,
        decision_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Retrieve recent chains for dashboard/API.

        Args:
            max_count:       Maximum chains to return
            track_id:        Filter to specific track
            decision_filter: Filter to specific decision type (e.g., "REROUTE")
        """
        with self._lock:
            chains = list(reversed(list(self._recent)))

        result = []
        for chain in chains:
            if track_id is not None and chain.track_id != track_id:
                continue
            if decision_filter and chain.final_decision != decision_filter:
                continue
            result.append(chain.to_dict())
            if len(result) >= max_count:
                break

        return result

    def get_latest_for_track(self, track_id: int) -> Optional[dict]:
        """Get the most recent chain for a specific track."""
        with self._lock:
            chain = self._by_track.get(track_id)
        return chain.to_dict() if chain else None

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_chains":     self._total,
                "decision_counts":  dict(self._decision_counts),
                "recent_count":     len(self._recent),
                "unique_tracks":    len(self._by_track),
            }

    def export_json(self, output_path: Path) -> None:
        """Export all chains to a JSON file for offline analysis."""
        with self._lock:
            chains = [c.to_dict() for c in self._recent]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chains, f, indent=2)
        logger.info(f"[ReasoningRegistry] Exported {len(chains)} chains → {output_path}")


# ==============================================================================
# GLOBAL REGISTRY SINGLETON
# ==============================================================================

_global_registry: Optional[ReasoningChainRegistry] = None


def get_reasoning_registry(maxlen: int = 200) -> ReasoningChainRegistry:
    """Get the global reasoning chain registry (creates on first call)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ReasoningChainRegistry(maxlen=maxlen)
    return _global_registry
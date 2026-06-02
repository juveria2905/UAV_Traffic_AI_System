"""
================================================================================
agentic_ai/memory/decision_memory.py — Decision Memory System
================================================================================

PURPOSE:
  Provides short-term and long-term memory for agent decisions.
  Enables the system to learn from past decisions, recognize patterns,
  and use historical context when making new decisions.

WHY THIS FILE EXISTS:
  Without memory, the agent system treats every frame as independent.
  A vehicle might have been REROUTED 5 frames ago — without memory,
  the agent doesn't know and might issue a conflicting EMERGENCY_STOP.
  Memory enables temporal reasoning: "Track #7 has been HIGH RISK
  for 3 consecutive frames — escalate response."

ARCHITECTURE:
  Short-term memory: last 50 decisions per track (rolling deque)
  Long-term memory:  aggregated statistics per track (lifetime)
  Pattern memory:    frequency analysis of decision types per track

CONNECTS TO:
  agent_system.py     → MonitorAgent retrieves history before deciding
  confidence_scorer.py → Uses success_rate from memory as input
  conflict_resolver.py → Uses historical decisions to break ties
  feedback_engine.py  → Writes outcome scores back to memory
  api.py             → /api/memory endpoint
  dashboard/app.py   → Memory Timeline tab
================================================================================
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# MEMORY RECORD
# ==============================================================================

@dataclass
class DecisionRecord:
    """
    One persisted agent decision with outcome tracking.

    Fields:
        track_id:       Vehicle this decision applies to
        frame_idx:      Frame when decision was made
        action:         Decision type ("REROUTE", "EMERGENCY_STOP", etc.)
        confidence:     Agent's confidence in this decision [0,1]
        agent_name:     Which agent made the decision
        ttc_at_time:    Time-to-collision when decision was made
        speed_at_time:  Vehicle speed when decision was made
        priority_score: Priority score when decision was made
        outcome_score:  Set later by FeedbackEngine (-1.0 to +1.0)
                        +1.0 = decision was correct and avoided issue
                         0.0 = neutral / not evaluated
                        -1.0 = decision was wrong (e.g., STOP when not needed)
        timestamp:      Unix timestamp
    """
    track_id:      int
    frame_idx:     int
    action:        str
    confidence:    float
    agent_name:    str
    ttc_at_time:   float   = 0.0
    speed_at_time: float   = 0.0
    priority_score: float  = 0.0
    density:       float   = 0.0
    outcome_score: float   = 0.0    # set by FeedbackEngine
    evaluated:     bool    = False  # True after FeedbackEngine scores it
    timestamp:     float   = field(default_factory=time.time)

    @property
    def age_frames(self) -> float:
        """Approximate age in frames (assumes ~50ms per frame)."""
        return (time.time() - self.timestamp) / 0.05

    def to_dict(self) -> dict:
        return {
            "track_id":      self.track_id,
            "frame_idx":     self.frame_idx,
            "action":        self.action,
            "confidence":    round(self.confidence, 4),
            "agent":         self.agent_name,
            "ttc":           round(self.ttc_at_time, 3),
            "speed":         round(self.speed_at_time, 3),
            "priority":      round(self.priority_score, 4),
            "density":       round(self.density, 4),
            "outcome_score": round(self.outcome_score, 4),
            "evaluated":     self.evaluated,
            "timestamp":     round(self.timestamp, 3),
        }


# ==============================================================================
# TRACK MEMORY — Per-vehicle accumulated knowledge
# ==============================================================================

@dataclass
class TrackMemory:
    """
    Accumulated memory for one tracked vehicle across its entire lifetime.

    SHORT-TERM: last 50 DecisionRecords (rolling buffer, for immediate context)
    LONG-TERM:  aggregated lifetime statistics

    PATTERN DETECTION:
        If EMERGENCY_STOP appears > 3 times in last 10 frames → REPEAT_OFFENDER
        If density correlation is high → CONGESTION_MAGNET
        If speed variance is high → ERRATIC_DRIVER
    """
    track_id: int
    short_term: deque = field(default_factory=lambda: deque(maxlen=50))

    # Long-term aggregates
    total_decisions:     int   = 0
    emergency_count:     int   = 0
    reroute_count:       int   = 0
    hold_count:          int   = 0
    monitor_count:       int   = 0
    total_outcome_score: float = 0.0
    evaluated_count:     int   = 0

    # Flags
    is_repeat_offender:    bool  = False
    is_erratic:            bool  = False
    is_congestion_magnet:  bool  = False

    def add_decision(self, record: DecisionRecord) -> None:
        """Add a new decision to short-term and update long-term stats."""
        self.short_term.append(record)
        self.total_decisions += 1

        action_counts = {
            "EMERGENCY_STOP": "emergency_count",
            "REROUTE":        "reroute_count",
            "HOLD":           "hold_count",
            "MONITOR":        "monitor_count",
        }
        attr = action_counts.get(record.action)
        if attr:
            setattr(self, attr, getattr(self, attr) + 1)

        self._update_flags()

    def update_outcome(self, frame_idx: int, outcome: float) -> bool:
        """
        Update the outcome score for a recent decision.
        FeedbackEngine calls this after observing what happened.

        Args:
            frame_idx: Frame of the decision to update
            outcome:   Score in [-1.0, +1.0]

        Returns:
            True if a matching record was found and updated.
        """
        for record in reversed(self.short_term):
            if record.frame_idx == frame_idx and not record.evaluated:
                record.outcome_score = outcome
                record.evaluated     = True
                self.total_outcome_score += outcome
                self.evaluated_count     += 1
                return True
        return False

    @property
    def success_rate(self) -> float:
        """
        Success rate based on evaluated decisions.
        Range [0, 1]: 1.0 = all decisions correct, 0.0 = all wrong.

        Derived from outcome_score: maps [-1.0, +1.0] → [0.0, 1.0]
        """
        if self.evaluated_count == 0:
            return 0.5   # unknown — neutral prior
        avg_outcome = self.total_outcome_score / self.evaluated_count
        return (avg_outcome + 1.0) / 2.0   # map [-1,+1] → [0,1]

    @property
    def recent_risk_rate(self) -> float:
        """
        Fraction of recent decisions that were high-risk (EMERGENCY or REROUTE).
        Computed over last 10 short-term records.
        """
        recent = list(self.short_term)[-10:]
        if not recent:
            return 0.0
        high_risk = sum(
            1 for r in recent
            if r.action in ("EMERGENCY_STOP", "REROUTE")
        )
        return high_risk / len(recent)

    def get_consecutive_action_count(self, action: str) -> int:
        """
        Count how many consecutive frames this track received `action`
        as the most recent decisions.

        Useful for escalation logic: if REROUTE for 5+ consecutive frames,
        upgrade to EMERGENCY_STOP.
        """
        count = 0
        for record in reversed(self.short_term):
            if record.action == action:
                count += 1
            else:
                break
        return count

    def _update_flags(self) -> None:
        """Update behavioral flags based on current history."""
        # Repeat offender: >2 emergencies in last 10 decisions
        recent_10 = list(self.short_term)[-10:]
        emerg_recent = sum(1 for r in recent_10 if r.action == "EMERGENCY_STOP")
        self.is_repeat_offender = emerg_recent > 2

        # Erratic: high speed variance in recent records
        speeds = [r.speed_at_time for r in recent_10 if r.speed_at_time > 0]
        if len(speeds) >= 5:
            import numpy as np
            mean_spd = float(np.mean(speeds))
            if mean_spd > 0.5:
                cv = float(np.std(speeds) / mean_spd)
                self.is_erratic = cv > 0.5
            else:
                self.is_erratic = False

        # Congestion magnet: frequently in high-density zones
        densities = [r.density for r in recent_10 if r.density > 0]
        if densities:
            import numpy as np
            self.is_congestion_magnet = float(np.mean(densities)) > 0.6

    def get_pattern_tags(self) -> List[str]:
        """Return list of behavioral pattern tags for dashboard display."""
        tags = []
        if self.is_repeat_offender:    tags.append("REPEAT_OFFENDER")
        if self.is_erratic:            tags.append("ERRATIC")
        if self.is_congestion_magnet:  tags.append("CONGESTION_MAGNET")
        if self.recent_risk_rate > 0.7: tags.append("CHRONIC_HIGH_RISK")
        return tags

    def get_recent_actions(self, n: int = 10) -> List[str]:
        """Return list of last n action strings."""
        return [r.action for r in list(self.short_term)[-n:]]

    def to_summary_dict(self) -> dict:
        return {
            "track_id":           self.track_id,
            "total_decisions":    self.total_decisions,
            "emergency_count":    self.emergency_count,
            "reroute_count":      self.reroute_count,
            "hold_count":         self.hold_count,
            "monitor_count":      self.monitor_count,
            "success_rate":       round(self.success_rate, 4),
            "recent_risk_rate":   round(self.recent_risk_rate, 4),
            "pattern_tags":       self.get_pattern_tags(),
            "recent_actions":     self.get_recent_actions(5),
            "repeat_offender":    self.is_repeat_offender,
            "erratic":            self.is_erratic,
            "congestion_magnet":  self.is_congestion_magnet,
        }


# ==============================================================================
# DECISION MEMORY — Main store
# ==============================================================================

class DecisionMemory:
    """
    Central decision memory store for the entire agent system.

    Manages per-track TrackMemory objects and provides global statistics.

    THREAD SAFETY: Uses threading.RLock for all state access.

    USAGE:
        memory = DecisionMemory()

        # Store a decision
        memory.store(DecisionRecord(
            track_id=7, frame_idx=221,
            action="REROUTE", confidence=0.91, agent_name="MonitorAgent-A",
            ttc_at_time=2.1, speed_at_time=8.3, priority_score=0.88,
        ))

        # Retrieve context before making new decision
        history = memory.get_track_history(7, n=5)
        # → [DecisionRecord, DecisionRecord, ...] most recent first

        # Check if track is a repeat offender
        if memory.is_repeat_offender(7):
            # Escalate response

        # Update outcome after observing what happened
        memory.update_outcome(track_id=7, frame_idx=221, outcome=+1.0)
    """

    def __init__(
        self,
        max_tracks:       int  = 500,
        prune_age_frames: int  = 5000,   # Remove tracks inactive for this many frames
        export_path:      Optional[Path] = None,
    ):
        """
        Args:
            max_tracks:       Maximum number of track memories to keep simultaneously.
                              Old inactive tracks are pruned when this is exceeded.
            prune_age_frames: Approximately how old (in frames) a track must be
                              before it's considered for pruning.
            export_path:      If set, periodically export memory to JSON.
        """
        self._lock           = threading.RLock()
        self._tracks:        Dict[int, TrackMemory] = {}
        self._max_tracks     = max_tracks
        self._prune_age      = prune_age_frames
        self._export_path    = export_path
        self._global_counts: Dict[str, int] = defaultdict(int)
        self._total_stored   = 0
        self._created_at     = time.time()

        logger.info(
            f"DecisionMemory initialized | "
            f"max_tracks={max_tracks}, prune_age={prune_age_frames} frames"
        )

    # ── STORAGE ───────────────────────────────────────────────────────────────

    def store(self, record: DecisionRecord) -> None:
        """
        Store a decision record in track memory.

        Creates a new TrackMemory for the track if first time seen.
        Automatically prunes oldest inactive tracks if at capacity.

        Args:
            record: DecisionRecord from an agent decision
        """
        with self._lock:
            if record.track_id not in self._tracks:
                if len(self._tracks) >= self._max_tracks:
                    self._prune_oldest()
                self._tracks[record.track_id] = TrackMemory(track_id=record.track_id)

            self._tracks[record.track_id].add_decision(record)
            self._global_counts[record.action] += 1
            self._total_stored += 1

        logger.debug(
            f"[Memory] Stored {record.action} for Track #{record.track_id} "
            f"@ frame {record.frame_idx} (conf={record.confidence:.2f})"
        )

    def store_batch(self, records: List[DecisionRecord]) -> None:
        """Store multiple records in a single lock acquisition (more efficient)."""
        with self._lock:
            for record in records:
                if record.track_id not in self._tracks:
                    if len(self._tracks) >= self._max_tracks:
                        self._prune_oldest()
                    self._tracks[record.track_id] = TrackMemory(track_id=record.track_id)
                self._tracks[record.track_id].add_decision(record)
                self._global_counts[record.action] += 1
                self._total_stored += 1

    # ── RETRIEVAL ─────────────────────────────────────────────────────────────

    def get_track_history(
        self,
        track_id: int,
        n:        int = 10,
    ) -> List[DecisionRecord]:
        """
        Get the last n decisions for a track, most recent first.

        Returns empty list if track not in memory.
        """
        with self._lock:
            tm = self._tracks.get(track_id)
            if tm is None:
                return []
            return list(reversed(list(tm.short_term)))[:n]

    def get_track_summary(self, track_id: int) -> Optional[dict]:
        """Get summary stats for a specific track."""
        with self._lock:
            tm = self._tracks.get(track_id)
            return tm.to_summary_dict() if tm else None

    def get_recent_action(self, track_id: int) -> Optional[str]:
        """Get the most recently stored action for this track."""
        with self._lock:
            tm = self._tracks.get(track_id)
            if tm and tm.short_term:
                return tm.short_term[-1].action
        return None

    def get_consecutive_count(self, track_id: int, action: str) -> int:
        """How many consecutive frames this track received `action`."""
        with self._lock:
            tm = self._tracks.get(track_id)
            if tm is None:
                return 0
            return tm.get_consecutive_action_count(action)

    def is_repeat_offender(self, track_id: int) -> bool:
        with self._lock:
            tm = self._tracks.get(track_id)
            return tm.is_repeat_offender if tm else False

    def is_erratic(self, track_id: int) -> bool:
        with self._lock:
            tm = self._tracks.get(track_id)
            return tm.is_erratic if tm else False

    def get_success_rate(self, track_id: int) -> float:
        """Get decision success rate for this track (0.5 if unknown)."""
        with self._lock:
            tm = self._tracks.get(track_id)
            return tm.success_rate if tm else 0.5

    def get_pattern_tags(self, track_id: int) -> List[str]:
        """Get behavioral pattern tags for this track."""
        with self._lock:
            tm = self._tracks.get(track_id)
            return tm.get_pattern_tags() if tm else []

    # ── OUTCOME UPDATES ───────────────────────────────────────────────────────

    def update_outcome(
        self,
        track_id:  int,
        frame_idx: int,
        outcome:   float,
    ) -> bool:
        """
        Update the outcome score for a past decision.
        Called by FeedbackEngine after observing actual results.

        Args:
            track_id:  Track to update
            frame_idx: Frame of the decision to update
            outcome:   Score in [-1.0, +1.0]
                       +1.0 = correct decision (avoided collision, reduced risk)
                        0.0 = neutral
                       -1.0 = wrong decision (unnecessary emergency, missed risk)

        Returns:
            True if the record was found and updated.
        """
        with self._lock:
            tm = self._tracks.get(track_id)
            if tm is None:
                return False
            return tm.update_outcome(frame_idx, outcome)

    # ── ANALYTICS ─────────────────────────────────────────────────────────────

    def get_all_summaries(self) -> List[dict]:
        """Return summaries for all tracked vehicles (for dashboard)."""
        with self._lock:
            return [tm.to_summary_dict() for tm in self._tracks.values()]

    def get_global_stats(self) -> dict:
        """Return system-wide memory statistics."""
        with self._lock:
            total_decisions = sum(
                tm.total_decisions for tm in self._tracks.values()
            )
            repeat_offenders = sum(
                1 for tm in self._tracks.values() if tm.is_repeat_offender
            )
            erratic_tracks = sum(
                1 for tm in self._tracks.values() if tm.is_erratic
            )
            avg_success = (
                sum(tm.success_rate for tm in self._tracks.values()) /
                max(len(self._tracks), 1)
            )

        return {
            "total_records_stored":   self._total_stored,
            "active_track_memories":  len(self._tracks),
            "total_decisions":        total_decisions,
            "global_action_counts":   dict(self._global_counts),
            "repeat_offenders":       repeat_offenders,
            "erratic_tracks":         erratic_tracks,
            "avg_success_rate":       round(avg_success, 4),
            "uptime_s":               round(time.time() - self._created_at, 1),
        }

    def get_decision_distribution(self) -> dict:
        """Return decision type distribution for charts."""
        with self._lock:
            return dict(self._global_counts)

    def get_high_risk_tracks(self, n: int = 10) -> List[dict]:
        """Return top-N tracks with highest recent risk rate."""
        with self._lock:
            summaries = [
                tm.to_summary_dict()
                for tm in self._tracks.values()
            ]
        return sorted(summaries, key=lambda x: x["recent_risk_rate"], reverse=True)[:n]

    def get_memory_timeline(
        self,
        track_id: int,
        max_records: int = 30,
    ) -> List[dict]:
        """
        Get timeline of decisions for one track (for dashboard visualization).
        Returns records sorted by frame_idx ascending.
        """
        with self._lock:
            tm = self._tracks.get(track_id)
            if tm is None:
                return []
            records = list(tm.short_term)

        records.sort(key=lambda r: r.frame_idx)
        return [r.to_dict() for r in records[-max_records:]]

    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def export_to_json(self, path: Optional[Path] = None) -> None:
        """Export all memory to JSON file for offline analysis."""
        output = path or self._export_path
        if output is None:
            logger.warning("[Memory] No export path configured.")
            return

        with self._lock:
            data = {
                "exported_at":  time.time(),
                "global_stats": self.get_global_stats(),
                "tracks":       [tm.to_summary_dict() for tm in self._tracks.values()],
            }

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[Memory] Exported {len(data['tracks'])} track memories → {output}")

    # ── PRUNING ───────────────────────────────────────────────────────────────

    def _prune_oldest(self) -> None:
        """
        Remove the oldest (least recently updated) track memory.
        Called automatically when at max_tracks capacity.

        WHY PRUNE:
        A 2-hour deployment processes thousands of tracks. Without pruning,
        memory would grow unbounded. We evict the oldest inactive track,
        which is least likely to reappear.
        """
        if not self._tracks:
            return

        # Find track with oldest most-recent decision
        oldest_tid = min(
            self._tracks.keys(),
            key=lambda tid: (
                self._tracks[tid].short_term[-1].timestamp
                if self._tracks[tid].short_term
                else 0.0
            )
        )
        del self._tracks[oldest_tid]
        logger.debug(f"[Memory] Pruned memory for Track #{oldest_tid}")

    def remove_track(self, track_id: int) -> None:
        """Explicitly remove a track's memory (e.g., when track is deleted)."""
        with self._lock:
            if track_id in self._tracks:
                del self._tracks[track_id]

    def clear(self) -> None:
        """Clear all memory. Use in testing or system reset."""
        with self._lock:
            self._tracks.clear()
            self._global_counts.clear()
            self._total_stored = 0
        logger.info("[Memory] All memory cleared.")


# ==============================================================================
# SINGLETON
# ==============================================================================

_global_memory: Optional[DecisionMemory] = None


def get_decision_memory(
    max_tracks:    int  = 500,
    export_path:   Optional[Path] = None,
) -> DecisionMemory:
    """Get or create the global decision memory singleton."""
    global _global_memory
    if _global_memory is None:
        _global_memory = DecisionMemory(
            max_tracks=max_tracks,
            export_path=export_path,
        )
    return _global_memory


def reset_decision_memory() -> None:
    """Reset global memory (use in tests)."""
    global _global_memory
    _global_memory = None
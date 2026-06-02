"""
Tests for agents/decision_memory.py
"""
import pytest
from agentic_ai.decision_memory import DecisionMemory
from config import DECISIONS


class TestDecisionMemory:

    def setup_method(self):
        self.mem = DecisionMemory(max_entries=100)

    # ── record & retrieve ──────────────────────────────────────────
    def test_record_single_entry(self):
        self.mem.record(1, "MONITOR", 0.3, 0.7, "test reason", frame=10)
        last = self.mem.last_decision(1)
        assert last is not None
        assert last["decision"] == "MONITOR"
        assert last["track_id"] == 1
        assert last["frame"] == 10

    def test_record_multiple_tracks(self):
        for tid in [1, 2, 3]:
            self.mem.record(tid, "HOLD", 0.5, 0.8, frame=1)
        assert len(self.mem.all_track_ids()) == 3

    def test_total_decisions_increments(self):
        self.mem.record(1, "MONITOR", 0.2, 0.7, frame=1)
        self.mem.record(1, "HOLD",    0.5, 0.8, frame=2)
        self.mem.record(2, "REROUTE", 0.7, 0.9, frame=3)
        assert self.mem.total_decisions == 3

    # ── historical_risk ────────────────────────────────────────────
    def test_historical_risk_zero_for_monitor_only(self):
        for f in range(10):
            self.mem.record(1, "MONITOR", 0.2, 0.7, frame=f)
        assert self.mem.historical_risk(1) == 0.0

    def test_historical_risk_one_for_emergency_only(self):
        for f in range(10):
            self.mem.record(1, "EMERGENCY", 0.95, 0.95, frame=f)
        assert self.mem.historical_risk(1) == 1.0

    def test_historical_risk_unknown_track_zero(self):
        assert self.mem.historical_risk(999) == 0.0

    def test_historical_risk_mixed_decisions(self):
        self.mem.record(1, "MONITOR",   0.2, 0.7, frame=1)
        self.mem.record(1, "EMERGENCY", 0.9, 0.9, frame=2)
        risk = self.mem.historical_risk(1)
        assert 0.0 < risk < 1.0

    # ── decision_counts ────────────────────────────────────────────
    def test_decision_counts_all_zero_initially(self):
        counts = self.mem.decision_counts()
        for d in DECISIONS:
            assert counts[d] == 0

    def test_decision_counts_correct(self):
        self.mem.record(1, "MONITOR",   0.2, 0.7, frame=1)
        self.mem.record(2, "REROUTE",   0.7, 0.9, frame=2)
        self.mem.record(3, "EMERGENCY", 0.9, 0.9, frame=3)
        counts = self.mem.decision_counts()
        assert counts["MONITOR"]   == 1
        assert counts["REROUTE"]   == 1
        assert counts["EMERGENCY"] == 1
        assert counts["HOLD"]      == 0

    # ── recent_global ──────────────────────────────────────────────
    def test_recent_global_respects_n(self):
        for i in range(30):
            self.mem.record(1, "MONITOR", 0.3, 0.7, frame=i)
        recent = self.mem.recent_global(n=10)
        assert len(recent) == 10

    def test_recent_global_ordered_ascending(self):
        for i in range(5):
            self.mem.record(1, "MONITOR", 0.3, 0.7, frame=i)
        recent = self.mem.recent_global(n=5)
        frames = [e["frame"] for e in recent]
        assert frames == sorted(frames)

    # ── track_summary ─────────────────────────────────────────────
    def test_track_summary_empty_for_unknown(self):
        s = self.mem.track_summary(999)
        assert s["total"] == 0

    def test_track_summary_counts_decisions(self):
        self.mem.record(5, "MONITOR", 0.3, 0.7, frame=1)
        self.mem.record(5, "HOLD",    0.5, 0.8, frame=2)
        self.mem.record(5, "MONITOR", 0.3, 0.7, frame=3)
        s = self.mem.track_summary(5)
        assert s["total"] == 3
        assert s["counts"]["MONITOR"] == 2
        assert s["counts"]["HOLD"] == 1
        assert s["last_decision"] == "MONITOR"

    # ── max_entries ring buffer ────────────────────────────────────
    def test_ring_buffer_does_not_exceed_max(self):
        mem = DecisionMemory(max_entries=10)
        for i in range(20):
            mem.record(1, "MONITOR", 0.3, 0.7, frame=i)
        assert len(mem.recent_global(n=100)) == 10
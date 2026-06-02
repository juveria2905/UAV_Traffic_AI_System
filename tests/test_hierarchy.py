"""
Integration tests for agents/hierarchy_manager.py
Tests the full agentic pipeline end-to-end.
"""
import pytest
import numpy as np
from agentic_ai.hierarchy_manager import HierarchyManager
from config import DECISIONS


@pytest.fixture
def hierarchy():
    return HierarchyManager()


@pytest.fixture
def tracks():
    return [
        {"id": 1, "bbox": [100, 100, 160, 150], "center": (130.0, 125.0),
         "class": "car", "confidence": 0.92, "velocity": (2.0, 1.5)},
        {"id": 2, "bbox": [200, 200, 260, 250], "center": (230.0, 225.0),
         "class": "car", "confidence": 0.85, "velocity": (-1.5, 2.0)},
        {"id": 3, "bbox": [500, 300, 560, 350], "center": (530.0, 325.0),
         "class": "person", "confidence": 0.71, "velocity": (0.5, 0.3)},
    ]


@pytest.fixture
def predictions(tracks):
    return {
        t["id"]: [(t["center"][0] + t["velocity"][0] * s,
                   t["center"][1] + t["velocity"][1] * s)
                  for s in range(1, 11)]
        for t in tracks
    }


@pytest.fixture
def speeds(tracks):
    return {t["id"]: float(np.hypot(*t["velocity"])) for t in tracks}


# ── Output structure ───────────────────────────────────────────────────────
class TestHierarchyOutputStructure:

    def test_returns_dict(self, hierarchy, tracks, predictions, speeds):
        out = hierarchy.process(tracks, predictions, [], speeds)
        assert isinstance(out, dict)

    def test_required_keys_present(self, hierarchy, tracks, predictions, speeds):
        out = hierarchy.process(tracks, predictions, [], speeds)
        required = ["frame", "decisions", "priority_results", "confidence_results",
                    "plan", "conflicts", "feedback_accuracy", "latency_ms",
                    "latency_history", "memory_summary", "total_decisions"]
        for key in required:
            assert key in out, f"Missing key: {key}"

    def test_decisions_keyed_by_track_id(self, hierarchy, tracks, predictions, speeds):
        out = hierarchy.process(tracks, predictions, [], speeds)
        for t in tracks:
            assert t["id"] in out["decisions"], f"No decision for track {t['id']}"

    def test_decisions_have_valid_values(self, hierarchy, tracks, predictions, speeds):
        out = hierarchy.process(tracks, predictions, [], speeds)
        for tid, d in out["decisions"].items():
            assert d["decision"] in DECISIONS, \
                f"Invalid decision '{d['decision']}' for track {tid}"

    def test_frame_counter_increments(self, hierarchy, tracks, predictions, speeds):
        out1 = hierarchy.process(tracks, predictions, [], speeds)
        out2 = hierarchy.process(tracks, predictions, [], speeds)
        assert out2["frame"] == out1["frame"] + 1

    def test_latency_ms_positive(self, hierarchy, tracks, predictions, speeds):
        out = hierarchy.process(tracks, predictions, [], speeds)
        assert out["latency_ms"] > 0

    def test_latency_history_grows(self, hierarchy, tracks, predictions, speeds):
        for _ in range(5):
            out = hierarchy.process(tracks, predictions, [], speeds)
        assert len(out["latency_history"]) == 5


# ── Decision logic ─────────────────────────────────────────────────────────
class TestHierarchyDecisionLogic:

    def test_emergency_when_collision_high_priority(self, hierarchy, tracks, speeds):
        # Simulate tracks very close together
        close_tracks = [
            {"id": 1, "bbox": [100, 100, 130, 130], "center": (115.0, 115.0),
             "class": "car", "confidence": 0.95, "velocity": (3.0, 0.0)},
            {"id": 2, "bbox": [120, 100, 150, 130], "center": (135.0, 115.0),
             "class": "car", "confidence": 0.95, "velocity": (-3.0, 0.0)},
        ]
        close_preds = {
            1: [(118, 115), (121, 115), (124, 115), (127, 115), (130, 115)],
            2: [(132, 115), (129, 115), (126, 115), (123, 115), (120, 115)],
        }
        close_colls = [{
            "track_a": 1, "track_b": 2, "ttc": 0.6,
            "distance_px": 20.0, "center": (125, 115),
            "bbox": [95, 95, 155, 135], "severity": "CRITICAL",
        }]
        close_speeds = {1: 3.0, 2: 3.0}

        # Process many frames to build confidence history and trigger escalation
        for _ in range(15):
            out = hierarchy.process(close_tracks, close_preds, close_colls, close_speeds)
        decisions = out["decisions"]
        # After 15 frames of high-risk, at least one should be REROUTE or EMERGENCY
        high_risk = [d for d in decisions.values()
                     if d["decision"] in ("REROUTE", "EMERGENCY")]
        assert len(high_risk) > 0, \
            f"Expected REROUTE or EMERGENCY after sustained collision, got: " \
            f"{[d['decision'] for d in decisions.values()]}"

    def test_monitor_for_isolated_slow_track(self):
        h = HierarchyManager()
        tracks = [{"id": 99, "bbox": [600, 500, 640, 530], "center": (620.0, 515.0),
                   "class": "car", "confidence": 0.7, "velocity": (0.1, 0.1)}]
        preds  = {99: [(620.1 + 0.1*s, 515.1 + 0.1*s) for s in range(1, 11)]}
        speeds = {99: 0.14}
        out    = h.process(tracks, preds, [], speeds)
        dec    = out["decisions"][99]["decision"]
        # Isolated, slow, no collision → should be MONITOR (or HOLD if goal override)
        assert dec in ("MONITOR", "HOLD")

    def test_empty_tracks_returns_empty_decisions(self, hierarchy):
        out = hierarchy.process([], {}, [], {})
        assert out["decisions"] == {}


# ── Memory integration ─────────────────────────────────────────────────────
class TestHierarchyMemoryIntegration:

    def test_total_decisions_increments_across_frames(self, hierarchy, tracks, predictions, speeds):
        for i in range(4):
            out = hierarchy.process(tracks, predictions, [], speeds)
        # 3 tracks × 4 frames = 12
        assert out["total_decisions"] == 3 * 4

    def test_memory_summary_covers_all_decisions(self, hierarchy, tracks, predictions, speeds):
        out = hierarchy.process(tracks, predictions, [], speeds)
        summary = out["memory_summary"]
        assert set(summary.keys()) == set(DECISIONS)
        assert sum(summary.values()) == len(tracks)


# ── Plan output ────────────────────────────────────────────────────────────
class TestHierarchyPlan:

    def test_plan_has_active_goals(self, hierarchy, tracks, predictions, speeds):
        out = hierarchy.process(tracks, predictions, [], speeds)
        assert len(out["plan"]["active_goals"]) >= 1

    def test_plan_has_system_mode(self, hierarchy, tracks, predictions, speeds):
        out = hierarchy.process(tracks, predictions, [], speeds)
        assert out["plan"]["system_mode"] in ("NORMAL", "HIGH_ALERT", "CRITICAL")
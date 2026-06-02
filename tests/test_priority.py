"""
Tests for agents/priority_engine.py
"""
import pytest
from agentic_ai.priority_engine import PriorityEngine
from config import PRIORITY_WEIGHTS, TTC_THRESHOLD


class TestPriorityEngine:

    def setup_method(self):
        self.engine = PriorityEngine()

    # ── output range ───────────────────────────────────────────────
    def test_output_in_zero_one_range(self):
        result = self.engine.compute(1, ttc=1.5, speed_px=10.0,
                                     nearby_count=3, history_risk=0.4)
        assert 0.0 <= result["priority_score"] <= 1.0

    def test_output_in_range_extreme_high(self):
        result = self.engine.compute(1, ttc=0.1, speed_px=100.0,
                                     nearby_count=10, history_risk=1.0)
        assert 0.0 <= result["priority_score"] <= 1.0

    def test_output_in_range_extreme_low(self):
        result = self.engine.compute(1, ttc=None, speed_px=0.0,
                                     nearby_count=0, history_risk=0.0)
        assert 0.0 <= result["priority_score"] <= 1.0

    # ── monotonicity ───────────────────────────────────────────────
    def test_lower_ttc_higher_priority(self):
        r_close = self.engine.compute(1, ttc=0.5,  speed_px=5.0, nearby_count=2, history_risk=0.2)
        r_far   = self.engine.compute(2, ttc=2.8,  speed_px=5.0, nearby_count=2, history_risk=0.2)
        assert r_close["priority_score"] > r_far["priority_score"]

    def test_higher_speed_higher_priority(self):
        r_fast = self.engine.compute(1, ttc=None, speed_px=35.0, nearby_count=0, history_risk=0.0)
        r_slow = self.engine.compute(2, ttc=None, speed_px=2.0,  nearby_count=0, history_risk=0.0)
        assert r_fast["priority_score"] > r_slow["priority_score"]

    def test_higher_density_higher_priority(self):
        r_dense  = self.engine.compute(1, ttc=None, speed_px=5.0, nearby_count=8, history_risk=0.0)
        r_sparse = self.engine.compute(2, ttc=None, speed_px=5.0, nearby_count=0, history_risk=0.0)
        assert r_dense["priority_score"] > r_sparse["priority_score"]

    def test_higher_history_risk_higher_priority(self):
        r_risky = self.engine.compute(1, ttc=None, speed_px=5.0, nearby_count=0, history_risk=1.0)
        r_safe  = self.engine.compute(2, ttc=None, speed_px=5.0, nearby_count=0, history_risk=0.0)
        assert r_risky["priority_score"] > r_safe["priority_score"]

    # ── no TTC ────────────────────────────────────────────────────
    def test_none_ttc_gives_zero_ttc_score(self):
        result = self.engine.compute(1, ttc=None, speed_px=0.0,
                                     nearby_count=0, history_risk=0.0)
        assert result["components"]["ttc_score"] == 0.0

    # ── components present ─────────────────────────────────────────
    def test_components_keys_present(self):
        result = self.engine.compute(1, ttc=1.0, speed_px=10.0,
                                     nearby_count=2, history_risk=0.3)
        for key in ["ttc_score", "speed_score", "density_score", "history_score"]:
            assert key in result["components"]

    def test_all_components_in_range(self):
        result = self.engine.compute(1, ttc=1.0, speed_px=10.0,
                                     nearby_count=2, history_risk=0.3)
        for k, v in result["components"].items():
            assert 0.0 <= v <= 1.0, f"Component {k}={v} out of [0,1]"

    # ── weights sum consistency ────────────────────────────────────
    def test_custom_weights(self):
        engine = PriorityEngine(weights={"ttc": 1.0, "speed": 0.0,
                                          "density": 0.0, "history": 0.0})
        r = engine.compute(1, ttc=0.5, speed_px=100.0,
                           nearby_count=10, history_risk=1.0)
        # With 100% weight on TTC, priority should be dominated by TTC
        assert r["priority_score"] > 0.8

    # ── track_id echo ──────────────────────────────────────────────
    def test_track_id_echoed(self):
        r = self.engine.compute(77, ttc=2.0, speed_px=5.0,
                                nearby_count=1, history_risk=0.2)
        assert r["track_id"] == 77

    # ── TTC at threshold ──────────────────────────────────────────
    def test_ttc_at_threshold_gives_zero_ttc_score(self):
        r = self.engine.compute(1, ttc=TTC_THRESHOLD, speed_px=0.0,
                                nearby_count=0, history_risk=0.0)
        assert r["components"]["ttc_score"] == pytest.approx(0.0, abs=1e-6)
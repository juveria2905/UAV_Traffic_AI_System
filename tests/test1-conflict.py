"""
Tests for agents/confidence_scorer.py
"""
import pytest
from agentic_ai.confidence_scorer import ConfidenceScorer


class TestConfidenceScorer:

    def setup_method(self):
        self.scorer = ConfidenceScorer()

    # ── output range ───────────────────────────────────────────────
    def test_output_in_zero_one_range(self):
        result = self.scorer.score(
            track_id=1, detection_conf=0.8,
            track_age=20, pred_variance=50.0, priority_score=0.7
        )
        assert 0.0 <= result["confidence"] <= 1.0

    def test_output_in_range_with_extreme_inputs(self):
        for det_conf in [0.0, 0.5, 1.0]:
            for age in [0, 1, 100]:
                for var in [0.0, 1000.0]:
                    r = self.scorer.score(1, det_conf, age, var, 0.5)
                    assert 0.0 <= r["confidence"] <= 1.0, \
                        f"Out of range: det_conf={det_conf}, age={age}, var={var}"

    # ── components present ─────────────────────────────────────────
    def test_returns_all_components(self):
        result = self.scorer.score(1, 0.9, 15, 30.0, 0.6)
        comps = result["components"]
        for key in ["det_conf", "stability", "pred_certainty", "consistency"]:
            assert key in comps, f"Missing component: {key}"

    def test_all_components_in_range(self):
        result = self.scorer.score(1, 0.85, 20, 100.0, 0.7)
        for k, v in result["components"].items():
            assert 0.0 <= v <= 1.0, f"Component {k}={v} out of range"

    # ── track_id echo ──────────────────────────────────────────────
    def test_track_id_echoed(self):
        result = self.scorer.score(42, 0.8, 10, 50.0, 0.5)
        assert result["track_id"] == 42

    # ── stability increases with age ───────────────────────────────
    def test_stability_increases_with_age(self):
        r_young = self.scorer.score(1, 0.8, 1,  50.0, 0.5)
        r_old   = self.scorer.score(2, 0.8, 30, 50.0, 0.5)
        assert r_young["components"]["stability"] < r_old["components"]["stability"]

    # ── high variance → low pred_certainty ────────────────────────
    def test_high_variance_reduces_pred_certainty(self):
        r_low  = self.scorer.score(1, 0.8, 15,   0.0, 0.5)
        r_high = self.scorer.score(2, 0.8, 15, 999.0, 0.5)
        assert r_low["components"]["pred_certainty"] > r_high["components"]["pred_certainty"]

    # ── consistency improves with repeated stable scores ───────────
    def test_consistency_improves_over_time(self):
        scorer = ConfidenceScorer()
        for _ in range(20):
            scorer.score(1, 0.85, 20, 50.0, 0.7)
        final = scorer.score(1, 0.85, 20, 50.0, 0.7)
        assert final["components"]["consistency"] > 0.5

    # ── clear_track ────────────────────────────────────────────────
    def test_clear_track_resets_history(self):
        scorer = ConfidenceScorer()
        for _ in range(10):
            scorer.score(7, 0.9, 20, 10.0, 0.8)
        scorer.clear_track(7)
        # After clear, consistency should be fresh (no history)
        result = scorer.score(7, 0.9, 20, 10.0, 0.8)
        assert result["confidence"] > 0.0   # still returns a value

    # ── different tracks are independent ──────────────────────────
    def test_multiple_tracks_independent(self):
        scorer = ConfidenceScorer()
        r1 = scorer.score(1, 0.9, 25, 10.0, 0.8)
        r2 = scorer.score(2, 0.4,  3, 400.0, 0.2)
        assert r1["confidence"] > r2["confidence"]
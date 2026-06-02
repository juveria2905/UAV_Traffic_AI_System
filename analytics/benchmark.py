"""
UAV System Benchmark – measures per-stage latency, FPS capability,
and agentic decision throughput. Outputs JSON + console table.

Usage:
    python analytics/benchmark.py [--frames 200] [--source data/sample.mp4]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
import statistics
import numpy as np
from config import VIDEO_SOURCE, OUTPUT_DIR
from utils.logger import get_logger

log = get_logger("Benchmark")


def run_benchmark(n_frames: int = 200, source=None) -> dict:
    from utils.frame_loader import FrameLoader
    from detection.yolo_detector import YOLODetector
    from detection.tracker       import DeepSORTTracker
    from prediction.trajectory_predictor import TrajectoryPredictor
    from prediction.collision_engine     import CollisionEngine
    from agents.hierarchy_manager        import HierarchyManager

    loader    = FrameLoader(source or VIDEO_SOURCE)
    detector  = YOLODetector()
    tracker   = DeepSORTTracker()
    predictor = TrajectoryPredictor()
    collision = CollisionEngine()
    hierarchy = HierarchyManager()

    timings = {
        "detect":    [],
        "track":     [],
        "predict":   [],
        "collision": [],
        "agent":     [],
        "total":     [],
    }
    n_tracks_per_frame = []
    n_collisions_per_frame = []

    log.info(f"Benchmarking {n_frames} frames…")
    processed = 0

    for _ in range(n_frames):
        ok, frame = loader.read()
        if not ok:
            break
        processed += 1
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        dets = detector.detect(frame)
        t2 = time.perf_counter()

        tracks = tracker.update(dets, frame)
        t3 = time.perf_counter()

        preds  = predictor.update(tracks)
        speeds = predictor.speed
        t4 = time.perf_counter()

        colls = collision.update(tracks, preds)
        t5 = time.perf_counter()

        agent_out = hierarchy.process(tracks, preds, colls, speeds)
        t6 = time.perf_counter()

        timings["detect"].append((t2 - t1) * 1000)
        timings["track"].append((t3 - t2) * 1000)
        timings["predict"].append((t4 - t3) * 1000)
        timings["collision"].append((t5 - t4) * 1000)
        timings["agent"].append((t6 - t5) * 1000)
        timings["total"].append((t6 - t0) * 1000)

        n_tracks_per_frame.append(len(tracks))
        n_collisions_per_frame.append(len(colls))

    loader.release()

    def stats(vals):
        if not vals:
            return {}
        return {
            "mean":  round(statistics.mean(vals), 2),
            "median":round(statistics.median(vals), 2),
            "stdev": round(statistics.stdev(vals) if len(vals) > 1 else 0, 2),
            "p95":   round(sorted(vals)[int(len(vals) * 0.95)], 2),
            "min":   round(min(vals), 2),
            "max":   round(max(vals), 2),
        }

    total_mean = statistics.mean(timings["total"]) if timings["total"] else 1
    result = {
        "frames_tested":        processed,
        "avg_fps":              round(1000 / max(total_mean, 1e-6), 2),
        "avg_latency_ms":       round(total_mean, 2),
        "p95_latency_ms":       stats(timings["total"]).get("p95", 0),
        "avg_tracks":           round(statistics.mean(n_tracks_per_frame), 2)
                                    if n_tracks_per_frame else 0,
        "avg_collisions":       round(statistics.mean(n_collisions_per_frame), 2)
                                    if n_collisions_per_frame else 0,
        "stage_timings_ms": {
            stage: stats(vals) for stage, vals in timings.items()
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ── Print table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  UAV SYSTEM BENCHMARK  ({processed} frames)")
    print("=" * 60)
    print(f"  Overall FPS:        {result['avg_fps']:.1f}")
    print(f"  Avg latency:        {result['avg_latency_ms']:.2f} ms")
    print(f"  P95 latency:        {result['p95_latency_ms']:.2f} ms")
    print(f"  Avg active tracks:  {result['avg_tracks']:.1f}")
    print(f"  Avg collisions:     {result['avg_collisions']:.1f}")
    print("-" * 60)
    for stage, s in result["stage_timings_ms"].items():
        if stage == "total":
            continue
        print(f"  {stage:<12}  mean={s.get('mean',0):6.2f}ms  "
              f"p95={s.get('p95',0):6.2f}ms")
    print("=" * 60 + "\n")

    # ── Save ──────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Benchmark results saved → {out_path}")

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int,   default=200,         help="Number of frames")
    p.add_argument("--source",             default=VIDEO_SOURCE, help="Video source")
    args = p.parse_args()
    run_benchmark(args.frames, args.source)
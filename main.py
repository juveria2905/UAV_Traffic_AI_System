"""
================================================================================
main.py — UAV Traffic Management System Main Processing Loop
================================================================================

Usage:
    python main.py [--source VIDEO_PATH] [--no-display] [--no-api] [--write]

Integration:
    - YOLODetector.predict_frame(frame, frame_idx) -> List[Detection]
    - DeepSORTTracker.update(frame, detections, frame_idx) -> Dict[int, TrackedObject]
    - TrajectoryPredictor.predict_all(tracks_dict) -> Dict[int, PredictedTrajectory]
    - CollisionEngine.run_frame(frame_idx, tracks_dict, predictions_dict) -> List[CollisionEvent]
    - HierarchicalAgentSystem.process_frame(frame_idx, tracks, collision_events, predictions) -> dict
    - SharedPipelineState.update(...) — pushes data to API layer
"""

import sys
import time
import threading
import argparse
import numpy as np
import cv2
import base64
from pathlib import Path
from typing import Dict, List, Any

from config import (
    VIDEO_SOURCE, WRITE_OUTPUT_VIDEO,
    DECISION_COLORS_BGR, DECISIONS,
    API_HOST, API_PORT,
)
from utils.logger       import get_logger
from utils.frame_loader import FrameLoader
from utils.video_writer import VideoWriter

from detection.yolo_detector import YOLODetector
from detection.tracker       import DeepSORTTracker, TrackedObject

from prediction.trajectory_predictor import TrajectoryPredictor
from prediction.collision_engine     import CollisionEngine, CollisionEvent

from agentic_ai.agent_system import HierarchicalAgentSystem

from backend.api import pipeline_state as store, push_frame
from backend.api import app as fastapi_app

log = get_logger("Main")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="UAV Traffic Management")
    p.add_argument("--source",     default=VIDEO_SOURCE, help="Video source")
    p.add_argument("--no-display", action="store_true",  help="Disable OpenCV window")
    p.add_argument("--no-api",     action="store_true",  help="Disable FastAPI server")
    p.add_argument("--write",      action="store_true",  help="Write annotated video")
    return p.parse_args()


# ── FastAPI background thread ─────────────────────────────────────────────────
def _run_api():
    import uvicorn
    # Allow socket reuse to prevent "address already in use" errors
    uvicorn.run(
        fastapi_app,
        host=API_HOST,
        port=API_PORT,
        log_level="warning",
        loop="uvloop" if False else "auto",  # Use asyncio (uvloop optional)
    )


# ── Wait for API to be ready ──────────────────────────────────────────────────
def _wait_for_api(timeout_sec: float = 10.0) -> bool:
    """Wait for FastAPI to become ready by polling health endpoint."""
    import requests
    start = time.time()
    api_url = f"http://{API_HOST}:{API_PORT}"

    while time.time() - start < timeout_sec:
        try:
            resp = requests.get(f"{api_url}/", timeout=0.5)
            if resp.ok:
                log.info("FastAPI is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass  # Connection refused — API not up yet
        except Exception:
            pass
        time.sleep(0.2)  # Increased backoff for slower systems

    log.warning(f"FastAPI did not respond within {timeout_sec}s, continuing anyway...")
    return False


# ── Convert TrackedObject dict to serialisable form ───────────────────────────
def _track_to_dict(track_id: int, track: TrackedObject, frame_idx: int) -> dict:
    """Serialise a TrackedObject for the API state store."""
    return track.to_dict(frame_idx)


# ── Convert CollisionEvent to serialisable form ───────────────────────────────
def _collision_to_dict(event: CollisionEvent) -> dict:
    return event.to_dict()


# ── Frame annotation ──────────────────────────────────────────────────────────
def annotate_frame(
    frame:       np.ndarray,
    tracks:      Dict[int, TrackedObject],
    decisions:   List[dict],           # list of AgentDecision.to_dict()
    collisions:  List[CollisionEvent],
    agent_state: dict,                 # system_state from agent output
    fps:         float,
    frame_idx:   int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out  = frame.copy()

    # Build decision lookup: track_id -> decision dict
    dec_by_track: Dict[int, dict] = {}
    for d in decisions:
        dec_by_track[d["track_id"]] = d

    # Draw trajectory trails
    for tid, track in tracks.items():
        if len(track.trail) < 2:
            continue
        pts = [(int(cx), int(cy)) for _, cx, cy in track.trail]
        for i in range(1, len(pts)):
            a = i / len(pts)
            col = (int(80 * a), int(180 * (1 - a)), int(255 * a))
            cv2.line(out, pts[i - 1], pts[i], col, 1)

    # Draw tracks with decision overlays
    for tid, track in tracks.items():
        if track.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
        dec    = dec_by_track.get(tid, {})
        action = dec.get("action", "MONITOR")
        conf   = dec.get("confidence", 0.0)
        col    = DECISION_COLORS_BGR.get(action, (0, 200, 0))

        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        # Label: Track ID | Class | Decision
        class_name = track.class_name if track.class_name else "unknown"
        label = f"#{tid} {class_name.upper()} | {action}"
        cv2.putText(out, label, (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Confidence score below
        cv2.putText(out, f"conf={conf:.2f}", (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        if track.center:
            cv2.circle(out, (int(track.center[0]), int(track.center[1])), 3, col, -1)

    # Draw collision alerts
    for ev in collisions:
        ta = tracks.get(ev.track_id_a)
        tb = tracks.get(ev.track_id_b)
        if ta and tb and ta.center and tb.center:
            p1 = (int(ta.center[0]), int(ta.center[1]))
            p2 = (int(tb.center[0]), int(tb.center[1]))
            risk = ev.risk_level.value if hasattr(ev.risk_level, "value") else str(ev.risk_level)
            col  = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255), "LOW": (0, 255, 255)}.get(risk, (0, 0, 255))
            cv2.line(out, p1, p2, col, 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(out, f"{risk} TTC={ev.ttc_seconds:.1f}s",
                        (mid[0] - 50, mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

    # HUD top bar
    n_trk  = len(tracks)
    emerg  = agent_state.get("emergency_stops", 0)
    rer    = agent_state.get("reroutes", 0)
    holds  = agent_state.get("holds", 0)
    mon    = agent_state.get("monitors", 0)

    cv2.rectangle(out, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(out,
                f"FPS:{fps:.1f}  Tracks:{n_trk}  Frame:{frame_idx}  "
                f"EMERGENCY:{emerg}  REROUTE:{rer}  HOLD:{holds}  MONITOR:{mon}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 0), 1)
    cv2.putText(out, f"Collisions:{len(collisions)}",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)

    return out


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Boot FastAPI
    if not args.no_api:
        api_thread = threading.Thread(target=_run_api, daemon=True)
        api_thread.start()
        log.info(f"FastAPI thread started, waiting for readiness...")
        _wait_for_api(timeout_sec=15.0)
        log.info(f"FastAPI available at http://{API_HOST}:{API_PORT}")

    # Initialise subsystems
    loader = FrameLoader(sequence_dir=args.source)
    if len(loader) == 0:
        log.error("No frames found. Exiting.")
        return

    frame_w, frame_h = loader.frame_size

    detector = YOLODetector()
    if not detector.load_model():
        log.error("YOLO model load failed. Exiting.")
        return

    tracker = DeepSORTTracker()
    if not tracker.initialize():
        log.error("Tracker init failed. Exiting.")
        return

    predictor = TrajectoryPredictor()
    collision  = CollisionEngine(fps=loader.fps)

    agent_system = HierarchicalAgentSystem(
        frame_width=frame_w,
        frame_height=frame_h,
    )

    writer = None
    if args.write:
        writer = VideoWriter(
            output_path="outputs/videos/uav_output.mp4",
            fps=loader.fps,
            frame_size=(frame_w, frame_h),
            codec="mp4v",
        )
        writer.open()

    fps_start   = time.perf_counter()
    fps_counter = 0
    fps_display = 0.0

    log.info("UAV Traffic Management System — RUNNING")

    try:
        for frame_idx, frame in loader:
            # ── Detection ────────────────────────────────────────────
            detections = detector.predict_frame(frame, frame_idx)

            # ── Tracking ─────────────────────────────────────────────
            # Signature: update(frame, detections, frame_idx) -> Dict[int, TrackedObject]
            tracks: Dict[int, TrackedObject] = tracker.update(
                frame, detections, frame_idx
            )

            # ── Prediction ───────────────────────────────────────────
            # predict_all(tracks_dict) -> Dict[int, PredictedTrajectory]
            predictions = predictor.predict_all(tracks)

            # ── Collision ────────────────────────────────────────────
            # run_frame(frame_idx, tracks, predictions) -> List[CollisionEvent]
            collision_events: List[CollisionEvent] = collision.run_frame(
                frame_idx, tracks, predictions
            )

            # ── Agentic AI ───────────────────────────────────────────
            # process_frame(frame_idx, tracks, collision_events, predictions) -> dict
            agent_out = agent_system.process_frame(
                frame_idx=frame_idx,
                tracks=tracks,
                collision_events=collision_events,
                predictions=predictions,
            )

            decisions_list: List[dict] = agent_out.get("decisions", [])
            system_state:   dict       = agent_out.get("system_state", {})

            # ── FPS ──────────────────────────────────────────────────
            fps_counter += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps_display  = fps_counter / elapsed
                fps_start    = time.perf_counter()
                fps_counter  = 0

            # ── Serialise for API ─────────────────────────────────────
            tracks_serial     = [_track_to_dict(tid, t, frame_idx) for tid, t in tracks.items()]
            collisions_serial = [_collision_to_dict(ev) for ev in collision_events]
            detections_serial = [d.to_dict() for d in detections]
            preds_serial      = [p.to_dict() for p in predictions.values()]

            # ── Push to API state ────────────────────────────────────
            store.update(
                frame=frame,
                detections=detections_serial,
                tracks=tracks_serial,
                predictions=preds_serial,
                collision_events=collisions_serial,
                agent_decisions=decisions_list,
                system_state={
                    **system_state,
                    "agent_latency_ms": agent_out.get("agent_latency_ms", 0),
                    "goal_summary":     agent_out.get("goal_summary", {}),
                    "memory_stats":     agent_out.get("memory_stats", {}),
                    "feedback_stats":   agent_out.get("feedback_stats", {}),
                    "city_state":       agent_out.get("city_state", {}),
                    "reasoning_chains": agent_out.get("reasoning_chains", []),
                },
                fps=fps_display,
                frame_idx=frame_idx,
                is_running=True,
            )

            # ── Annotate ─────────────────────────────────────────────
            ann = annotate_frame(
                frame, tracks, decisions_list,
                collision_events, system_state,
                fps_display, frame_idx,
            )
            push_frame(ann)

            if writer:
                writer.write(ann)

            if not args.no_display:
                cv2.imshow("UAV Traffic AI — Q to quit", ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        if writer:
            writer.release()
        agent_system.save_logs(Path("outputs"))
        cv2.destroyAllWindows()
        store.update(is_running=False)
        log.info("System stopped cleanly.")


if __name__ == "__main__":
    main()
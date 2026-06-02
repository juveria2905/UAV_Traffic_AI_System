"""
================================================================================
backend/api.py — FastAPI REST + WebSocket Backend
================================================================================

DAY 6 SUBSYSTEM — BACKEND

═══════════════════════════════════════════════════════════════════════════════
CONCEPT: WHY FASTAPI?
═══════════════════════════════════════════════════════════════════════════════

FastAPI is a modern Python web framework that:
  - Automatically generates API documentation (Swagger UI at /docs)
  - Uses Python type hints for request/response validation
  - Supports async/await natively (non-blocking I/O)
  - Supports WebSockets for real-time bidirectional communication

OUR ARCHITECTURE:
  Pipeline (main.py)
    ↓ pushes data to SharedPipelineState
  FastAPI (this file)
    ↓ reads from SharedPipelineState
  REST endpoints → Streamlit polls for latest data
  WebSocket → Streamlit receives frame stream in real time

SHARED STATE PATTERN:
  We use a global singleton object (SharedPipelineState) to pass data
  from the processing pipeline to the API endpoints.
  
  Why? FastAPI runs in the same process, so Python objects are shared
  in memory. The pipeline writes to state; API reads from state.
  No need for Redis/database for MVP.

WEBSOCKET vs REST:
  REST (HTTP):     Request → Response. Dashboard asks → Server answers.
                   Good for: current stats, history, one-time data.
  
  WebSocket:       Persistent bidirectional connection.
                   Server PUSHES data to client without being asked.
                   Good for: live video frames, real-time alerts.
                   
  We use both:
    - REST for stats, decisions history
    - WebSocket for live frame stream

BASE64 FRAME ENCODING:
  We can't send raw numpy arrays over HTTP/WebSocket.
  Solution: encode frame as JPEG → base64 string → JSON.
  Client decodes base64 → display as <img src="data:image/jpeg;base64,..."/>
================================================================================
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
import threading

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import API_CONFIG, LOG_CONFIG
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


# ==============================================================================
# SHARED PIPELINE STATE
# ==============================================================================

class SharedPipelineState:
    """
    Thread-safe shared state between the pipeline and the API.

    The main pipeline (running in a thread) WRITES to this object.
    FastAPI endpoint handlers READ from this object.

    THREAD SAFETY:
    We use threading.Lock() to prevent race conditions when both the
    pipeline thread and the API thread access the same data simultaneously.

    A race condition example:
      Thread A is writing new_detections list (halfway done)
      Thread B reads the list (gets incomplete data)
      → Corrupted read

    With a lock:
      Thread A acquires lock → writes → releases lock
      Thread B waits for lock → acquires → reads complete data
    """

    def __init__(self, max_frame_buffer: int = 5):
        self._lock = threading.Lock()

        # Latest annotated frame (numpy array BGR)
        self._current_frame: Optional[np.ndarray] = None

        # Latest detections
        self._detections: List[dict] = []

        # Latest track states
        self._tracks: List[dict] = []

        # Latest trajectory predictions
        self._predictions: List[dict] = []

        # Latest collision events
        self._collision_events: List[dict] = []

        # Latest agent decisions
        self._agent_decisions: List[dict] = []

        # System state summary
        self._system_state: dict = {}

        # FPS
        self._fps: float = 0.0

        # Frame index
        self._frame_idx: int = 0

        # Pipeline running flag (initialize to True so API is ready on startup)
        self._is_running: bool = True

        # Alert queue: recent HIGH risk events (for dashboard sidebar)
        self._alert_queue: deque = deque(maxlen=50)

        # Startup timestamp
        self._startup_time: float = time.time()

    def update(
        self,
        frame:             Optional[np.ndarray] = None,
        detections:        Optional[List[dict]] = None,
        tracks:            Optional[List[dict]] = None,
        predictions:       Optional[List[dict]] = None,
        collision_events:  Optional[List[dict]] = None,
        agent_decisions:   Optional[List[dict]] = None,
        system_state:      Optional[dict] = None,
        fps:               float = 0.0,
        frame_idx:         int = 0,
        is_running:        bool = True,
    ) -> None:
        """Update shared state with latest pipeline output. Thread-safe."""
        with self._lock:
            if frame is not None:
                self._current_frame = frame.copy()
            if detections is not None:
                self._detections = detections
            if tracks is not None:
                self._tracks = tracks
            if predictions is not None:
                self._predictions = predictions
            if collision_events is not None:
                self._collision_events = collision_events
                # Push HIGH risk events to alert queue
                for ev in collision_events:
                    if ev.get("risk_level") == "HIGH":
                        ev["timestamp"] = time.time()
                        self._alert_queue.append(ev)
            if agent_decisions is not None:
                self._agent_decisions = agent_decisions
            if system_state is not None:
                self._system_state = system_state
            self._fps       = fps
            self._frame_idx = frame_idx
            self._is_running = is_running

    def get_snapshot(self) -> dict:
        """Return a full snapshot of current state. Thread-safe."""
        with self._lock:
            return {
                "frame_idx":       self._frame_idx,
                "fps":             round(self._fps, 1),
                "is_running":      self._is_running,
                "detections":      list(self._detections),
                "tracks":          list(self._tracks),
                "predictions":     list(self._predictions),
                "collision_events": list(self._collision_events),
                "agent_decisions": list(self._agent_decisions),
                "system_state":    dict(self._system_state),
                "alert_queue":     list(self._alert_queue),
            }

    def get_current_frame_jpeg(self, quality: int = 75) -> Optional[bytes]:
        """
        Encode current frame as JPEG bytes. Thread-safe.
        
        JPEG compression tradeoffs:
          quality=95 → large file, high quality (~200KB per frame)
          quality=75 → balanced (~50KB per frame) ← we use this
          quality=50 → small file, noticeable artifacts (~25KB)
        
        For real-time streaming, 75 is the sweet spot.
        """
        with self._lock:
            if self._current_frame is None:
                return None
            frame_copy = self._current_frame.copy()

        # Encode to JPEG bytes
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode(".jpg", frame_copy, encode_params)

        if not success:
            return None

        return buffer.tobytes()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def set_running(self, running: bool) -> None:
        """Explicitly set the running flag."""
        with self._lock:
            self._is_running = running
            logger.debug(f"Pipeline running flag set to: {running}")


# Global singleton — shared between pipeline and API
pipeline_state = SharedPipelineState()
# Compatibility alias for main.py
state_store = pipeline_state

def push_frame(frame):
    """Store latest frame for dashboard."""
    with pipeline_state._lock:
        pipeline_state._current_frame = frame.copy()

# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="UAV Traffic AI — Backend API",
    description="Real-time UAV traffic management system API",
    version="1.0.0",
)

# CORS Middleware
# Allows Streamlit (running at localhost:8501) to call this API (localhost:8000)
# Without CORS, browsers block cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# WEBSOCKET CONNECTION MANAGER
# ==============================================================================

class ConnectionManager:
    """
    Manages active WebSocket connections.
    
    Multiple dashboard tabs can connect simultaneously.
    We broadcast to ALL connected clients every time a new frame arrives.
    
    BROADCAST PATTERN:
      New frame arrives → encode as base64 JPEG → send to all active connections
      If a client disconnects mid-send → remove from active list
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def broadcast(self, data: dict) -> None:
        """Send data to all connected WebSocket clients."""
        if not self.active_connections:
            return

        message = json.dumps(data)
        disconnected = []

        async with self._lock:
            connections_copy = list(self.active_connections)

        for ws in connections_copy:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        # Remove disconnected clients
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    if ws in self.active_connections:
                        self.active_connections.remove(ws)


ws_manager = ConnectionManager()


# ==============================================================================
# REST ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status":  "running",
        "service": "UAV Traffic AI Backend",
        "version": "1.0.0",
    }


@app.get("/api/status")
async def get_status():
    """
    Returns current pipeline status.
    Called by dashboard to check if pipeline is running.
    """
    snapshot = pipeline_state.get_snapshot()
    return {
        "is_running":     snapshot["is_running"],
        "frame_idx":      snapshot["frame_idx"],
        "fps":            snapshot["fps"],
        "active_tracks":  len(snapshot["tracks"]),
        "detections":     len(snapshot["detections"]),
        "collision_risks": len(snapshot["collision_events"]),
        "timestamp":      time.time(),
    }


@app.get("/api/detections")
async def get_detections():
    """Returns latest frame's detection results."""
    snapshot = pipeline_state.get_snapshot()
    return {
        "frame_idx":  snapshot["frame_idx"],
        "count":      len(snapshot["detections"]),
        "detections": snapshot["detections"],
    }


@app.get("/api/tracks")
async def get_tracks():
    """Returns currently active tracked objects."""
    snapshot = pipeline_state.get_snapshot()
    return {
        "frame_idx": snapshot["frame_idx"],
        "count":     len(snapshot["tracks"]),
        "tracks":    snapshot["tracks"],
    }


@app.get("/api/predictions")
async def get_predictions():
    """Returns trajectory predictions for active tracks."""
    snapshot = pipeline_state.get_snapshot()
    return {
        "frame_idx":   snapshot["frame_idx"],
        "count":       len(snapshot["predictions"]),
        "predictions": snapshot["predictions"],
    }


@app.get("/api/collisions")
async def get_collisions():
    """Returns current collision risk events."""
    snapshot = pipeline_state.get_snapshot()
    return {
        "frame_idx":       snapshot["frame_idx"],
        "count":           len(snapshot["collision_events"]),
        "collision_events": snapshot["collision_events"],
        "alerts":          list(pipeline_state._alert_queue),
    }


@app.get("/api/decisions")
async def get_decisions():
    """Returns latest agent decisions."""
    snapshot = pipeline_state.get_snapshot()
    return {
        "frame_idx":      snapshot["frame_idx"],
        "count":          len(snapshot["agent_decisions"]),
        "decisions":      snapshot["agent_decisions"],
        "system_state":   snapshot["system_state"],
    }


@app.get("/api/frame")
async def get_frame():
    """
    Returns latest annotated frame as base64-encoded JPEG.
    Streamlit can display this as an image.
    
    RESPONSE FORMAT:
      {
        "frame_idx": 42,
        "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."
      }
    """
    jpeg_bytes = pipeline_state.get_current_frame_jpeg(quality=75)

    if jpeg_bytes is None:
        raise HTTPException(status_code=404, detail="No frame available yet.")

    b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_str}"

    return {
        "frame_idx": pipeline_state._frame_idx,
        "image_b64": data_url,
    }


@app.get("/api/snapshot")
async def get_full_snapshot():
    """Returns complete system state snapshot (without frame image)."""
    return pipeline_state.get_snapshot()


# ==============================================================================
# WEBSOCKET ENDPOINT
# ==============================================================================

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint for live frame streaming.
    
    HOW IT WORKS:
      1. Client connects to ws://localhost:8000/ws/live
      2. Server accepts connection and adds to active_connections
      3. The background task (push_frames_loop) periodically sends frames
         to all connected clients
      4. On disconnect (client closes tab), we remove from active_connections
    
    MESSAGE FORMAT (JSON):
      {
        "type":        "frame",
        "frame_idx":   42,
        "fps":         18.5,
        "image_b64":   "data:image/jpeg;base64,...",
        "tracks":      [...],
        "collisions":  [...],
        "decisions":   [...],
        "system_state": {...}
      }
    """
    await ws_manager.connect(websocket)
    logger.info("New WebSocket client connected for live stream.")

    try:
        while True:
            # Keep connection alive by receiving (client can send messages too)
            # timeout allows us to periodically push data
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                # Client can send commands: {"action": "pause"} etc.
                logger.debug(f"WS received: {data}")
            except asyncio.TimeoutError:
                pass

            # Push current state to THIS client
            jpeg_bytes = pipeline_state.get_current_frame_jpeg(quality=70)
            snapshot   = pipeline_state.get_snapshot()

            payload = {
                "type":         "frame",
                "frame_idx":    snapshot["frame_idx"],
                "fps":          snapshot["fps"],
                "image_b64":    (
                    f"data:image/jpeg;base64,"
                    f"{base64.b64encode(jpeg_bytes).decode()}"
                ) if jpeg_bytes else None,
                "tracks":       snapshot["tracks"][:20],    # Limit payload size
                "collisions":   snapshot["collision_events"],
                "decisions":    snapshot["agent_decisions"][:20],
                "system_state": snapshot["system_state"],
            }

            try:
                await websocket.send_text(json.dumps(payload))
            except Exception:
                break

            # ~20 FPS push rate
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)


@app.get("/state")
async def dashboard_state():
    snapshot = pipeline_state.get_snapshot()

    return {
        "frame": snapshot["frame_idx"],
        "fps": snapshot["fps"],
        "tracks": snapshot["tracks"],
        "collisions": snapshot["collision_events"],
        "decisions": {},
        "priority_results": {},
        "confidence_results": {},
        "memory_summary": {},
        "total_decisions": 0,
        "feedback_accuracy": 1.0,
        "fps_history": [],
        "latency_history": [],
        "plan": {},
        "conflicts": [],
        "system_status": (
            "RUNNING" if snapshot["is_running"] else "STOPPED"
        ),
    }


@app.get("/memory")
async def get_memory():
    return {
        "recent": []
    }








# ==============================================================================
# SERVER LAUNCH
# ==============================================================================

def start_api_server(host: str = None, port: int = None, reload: bool = False):
    """Start the FastAPI server with uvicorn."""
    host = host or API_CONFIG.host
    port = port or API_CONFIG.port

    logger.info(f"Starting FastAPI server at http://{host}:{port}")
    logger.info(f"API docs at: http://{host}:{port}/docs")

    uvicorn.run(
        "backend.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",   # Suppress uvicorn's verbose access logs
    )


if __name__ == "__main__":
    start_api_server()
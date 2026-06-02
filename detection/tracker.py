"""
================================================================================
detection/tracker.py — DeepSORT Multi-Object Tracking
================================================================================

DAY 2 SUBSYSTEM — TRACKING

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 1: WHAT IS TRACKING AND WHY DO WE NEED IT?
═══════════════════════════════════════════════════════════════════════════════

Detection answers: "WHAT is in this frame, and WHERE?"
Tracking answers:  "IS THIS THE SAME OBJECT as last frame?"

Without tracking:
  Frame 1: "Car at position (100, 200)"
  Frame 2: "Car at position (105, 205)"
  Frame 3: "Car at position (110, 210)"
  → We have NO idea these are the same car.

With tracking:
  Frame 1: "Car #7 at (100, 200)"
  Frame 2: "Car #7 at (105, 205)"  ← SAME car, different position
  Frame 3: "Car #7 at (110, 210)"
  → Now we know the car MOVED. We can calculate velocity, predict future position.

WHY TRACKING IS ESSENTIAL BEFORE TRAJECTORY PREDICTION:
  Trajectory = history of positions for ONE object over time
  You can only build a trajectory if you know which detection belongs to which object
  That identity-assignment is EXACTLY what tracking does.

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 2: HOW DEEPSORT WORKS — THE INTUITION
═══════════════════════════════════════════════════════════════════════════════

DeepSORT = Deep Learning + SORT (Simple Online and Realtime Tracking)

It solves two problems:
  PROBLEM A: Where will object #7 be in the NEXT frame?
  PROBLEM B: Which detection in the next frame IS object #7?

STEP-BY-STEP PER FRAME:
  
  Step 1 — PREDICT (Kalman Filter)
    Before seeing new detections, DeepSORT predicts where each tracked
    object SHOULD be based on its last known velocity.
    "Car #7 was moving right at 5px/frame → it should be at x+5 now"
  
  Step 2 — DETECT (YOLOv8)
    Get fresh detections from the current frame.
  
  Step 3 — MATCH (Hungarian Algorithm)
    Compare predictions to actual detections.
    Use two signals to match:
      a) IoU (Intersection over Union) — spatial overlap
      b) Appearance embedding — does this detection LOOK like track #7?
    This is the "Deep" in DeepSORT — a neural net extracts appearance features.
  
  Step 4 — UPDATE (Kalman Filter)
    For matched pairs: update the track's state with the real detection.
    For unmatched tracks: keep predicting (object might be temporarily hidden).
    For unmatched detections: create a NEW track.
  
  Step 5 — ASSIGN IDs
    New tracks get new IDs. Existing tracks keep their IDs.
    Track is deleted if object disappears for max_age frames.

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 3: KALMAN FILTER — THE INTUITION
═══════════════════════════════════════════════════════════════════════════════

The Kalman Filter is a mathematical way to say:
  "I have a NOISY measurement (detection from YOLOv8).
   I have a PHYSICS MODEL (objects move smoothly, not randomly).
   The truth is somewhere BETWEEN my model's prediction and my measurement."

Think of it like this:
  Your GPS says you're at position X (noisy measurement).
  Your last speed + direction says you should be at Y (physics model).
  Kalman says: "The truth is 0.6*X + 0.4*Y" (weighted average based on noise).

STATE VECTOR tracked by Kalman for each object:
  [x_center, y_center, width, height, vx, vy, vw, vh]
  Where vx, vy = velocity in x and y directions
  The filter tracks both POSITION and VELOCITY.

WHY THIS MATTERS:
  Even when YOLOv8 fails to detect an object for 2-3 frames (occlusion),
  the Kalman Filter keeps predicting its position based on velocity.
  This prevents ID loss during brief disappearances.

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 4: TRACKING FAILURES — WHAT CAN GO WRONG
═══════════════════════════════════════════════════════════════════════════════

  ID SWITCH: Object #7 suddenly gets labeled #23 because two objects
  crossed and the matcher got confused.

  TRACK FRAGMENTATION: Object disappears and reappears → gets a new ID.
  Now you have #7 for frames 1-50 and #31 for frames 52-100, same car.

  GHOST TRACKS: A track keeps predicting even though object is gone.
  Deleted after max_age frames without a match.

  MERGE: Two close objects treated as one track.

DeepSORT minimizes these with appearance embeddings.
But they still happen — understanding failure modes helps debugging.

═══════════════════════════════════════════════════════════════════════════════
LIBRARIES USED TODAY
═══════════════════════════════════════════════════════════════════════════════

deep_sort_realtime:
  - Python wrapper around the DeepSORT algorithm
  - Key class: DeepSort
  - Input: list of ([x,y,w,h], confidence, class_id) per frame
  - Output: list of Track objects with .track_id, .to_ltrb(), .det_class

collections.deque:
  - A double-ended queue with a maxlen argument
  - When full, oldest item is automatically dropped (perfect for trail history)
  - O(1) append and pop from both ends (unlike list which is O(n) for left ops)
  - We use deque(maxlen=50) to keep the last 50 positions as a trajectory trail

numpy:
  - Used for all bounding box math
  - Converting between [x,y,w,h] and [x1,y1,x2,y2] formats
  - Computing centers, distances

OpenCV:
  - cv2.polylines() to draw the trajectory trail
  - cv2.circle() for current position dot
  - cv2.putText() for track ID labels
================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
import time
import csv

# DeepSORT: the tracking algorithm
# DeepSort class from deep_sort_realtime package
from deep_sort_realtime.deepsort_tracker import DeepSort

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    YOLO_MODEL,
    UAVDT_SEQ_DIR,
    OUTPUT_VIDEO_DIR,
    OUTPUT_CSV_DIR,
    DETECTION_CONFIG,
    TRACKING_CONFIG,
    VIDEO_CONFIG,
    LOG_CONFIG,
)
from utils.logger import get_logger, setup_logging
from utils.frame_loader import FrameLoader
from utils.video_writer import VideoWriter
from detection.yolo_detector import YOLODetector, Detection

logger = get_logger(__name__)


# ==============================================================================
# TRACK DATA STRUCTURE
# ==============================================================================

class TrackedObject:
    """
    Represents a single tracked object with persistent ID and trajectory history.
    
    WHY WE WRAP DeepSORT's TRACK OBJECT:
    DeepSORT gives us raw track objects. We wrap them in our own class to:
      - Add trajectory history (DeepSORT doesn't keep position history)
      - Add class name lookups
      - Provide clean serialization for CSV/API output
      - Decouple our system from DeepSORT internals (easier to swap later)
    """

    def __init__(self, track_id: int, class_id: int, class_name: str, max_trail: int = 50):
        self.track_id   = track_id
        self.class_id   = class_id
        self.class_name = class_name

        # deque with maxlen automatically drops oldest entries when full
        # This gives us a rolling window of the last max_trail positions
        # Each entry is (frame_idx, cx, cy) — center x and center y
        self.trail: deque = deque(maxlen=max_trail)

        # Current bounding box in (x1,y1,x2,y2) format
        self.bbox_xyxy: Optional[Tuple[float,float,float,float]] = None

        # Current confidence score
        self.confidence: float = 0.0

        # Frame index when this track was first created
        self.birth_frame: int = 0

        # Frame index of last update
        self.last_frame: int = 0

        # Age in frames (how long this track has existed)
        self.age: int = 0

    def update(
        self,
        frame_idx: int,
        bbox_xyxy: Tuple[float,float,float,float],
        confidence: float,
    ) -> None:
        """Update track with latest detection."""
        self.bbox_xyxy  = bbox_xyxy
        self.confidence = confidence
        self.last_frame = frame_idx
        self.age        = frame_idx - self.birth_frame

        # Compute center from bounding box and add to trail
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.trail.append((frame_idx, cx, cy))

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        """Current center position (cx, cy). None if no bbox yet."""
        if not self.trail:
            return None
        _, cx, cy = self.trail[-1]
        return (cx, cy)

    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        """
        Estimated velocity (vx, vy) pixels per frame.
        Computed from last 2 trail points.
        Returns None if insufficient history.
        """
        if len(self.trail) < 2:
            return None
        _, cx1, cy1 = self.trail[-2]
        _, cx2, cy2 = self.trail[-1]
        return (cx2 - cx1, cy2 - cy1)

    def to_dict(self, frame_idx: int) -> dict:
        """Serialize to dict for CSV/API output."""
        cx, cy = self.center if self.center else (0, 0)
        vx, vy = self.velocity if self.velocity else (0, 0)
        bbox   = self.bbox_xyxy if self.bbox_xyxy else (0, 0, 0, 0)
        return {
            "frame_idx":  frame_idx,
            "track_id":   self.track_id,
            "class_name": self.class_name,
            "x1": round(bbox[0], 2),
            "y1": round(bbox[1], 2),
            "x2": round(bbox[2], 2),
            "y2": round(bbox[3], 2),
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "confidence": round(self.confidence, 4),
            "age_frames": self.age,
        }

    def __repr__(self) -> str:
        return (
            f"TrackedObject(id={self.track_id}, class={self.class_name}, "
            f"age={self.age}, trail_len={len(self.trail)})"
        )


# ==============================================================================
# DEEPSORT TRACKER
# ==============================================================================

class DeepSORTTracker:
    """
    Wraps the DeepSORT algorithm and integrates it with our YOLOv8 detections.
    
    RESPONSIBILITIES:
      - Initialize and configure DeepSORT
      - Convert YOLOv8 detections to DeepSORT input format
      - Run tracking per frame
      - Maintain TrackedObject registry
      - Annotate frames with IDs and trails
      - Export trajectory data to CSV
    
    DATA FLOW PER FRAME:
      YOLOv8 detections (Detection objects)
          → convert to DeepSORT format ([x,y,w,h], conf, class)
          → DeepSORT.update_tracks()
          → DeepSORT returns Track objects with assigned IDs
          → We update our TrackedObject registry
          → We annotate the frame
    """

    def __init__(self, config: dict = None):
        self.cfg = config or TRACKING_CONFIG
        self._tracker: Optional[DeepSort] = None

        # Registry of all active tracked objects: {track_id: TrackedObject}
        self._tracks: Dict[int, TrackedObject] = {}

        # Class name lookup from config
        self._class_names = getattr(DETECTION_CONFIG,"class_names",{})

        self._class_colors = getattr(DETECTION_CONFIG,"class_colors",{})

        # Statistics
        self._total_tracks_created = 0
        self._frame_count = 0

        # CSV export buffer
        self._csv_rows: List[dict] = []

    def initialize(self) -> bool:
        """
        Initialize the DeepSORT tracker.
        
        DeepSort PARAMETERS EXPLAINED:
        
        max_age:
          How many frames a track survives without a matching detection.
          If a car goes behind a building for 30 frames and we set max_age=30,
          it keeps its ID when it reappears. Set too low → frequent ID loss.
          Set too high → ghost tracks linger too long.
        
        n_init:
          A new track must be confirmed by detections in n_init consecutive
          frames before getting an ID. Prevents assigning IDs to false positives.
          n_init=3 means: a blob must appear 3 frames in a row to become a track.
        
        nms_max_overlap:
          NMS at the tracker level. Removes duplicate tracks for same object.
        
        max_cosine_distance:
          For appearance matching. Lower = stricter appearance matching.
          The appearance model computes a feature vector for each detection.
          If cosine distance between track's stored appearance and new detection
          exceeds this threshold, they won't be matched on appearance.
        
        embedder:
          The neural network that extracts appearance features.
          'mobilenet' is fast and good enough for vehicles.
          Options: 'mobilenet', 'torchreid', 'clip_RN50'
        """
        max_age=self.cfg.max_age
        min_hits=self.cfg.min_hits
        iou_thresh=self.cfg.iou_threshold

        try:
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=min_hits,
                nms_max_overlap=1.0,
                max_cosine_distance=0.5,
                nn_budget=None,          # No limit on appearance memory per track
                override_track_class=None,
                embedder="mobilenet",    # Fast appearance extractor
                half=False,             # Use full precision (more accurate)
                bgr=True,               # Our frames are BGR (OpenCV default)
                embedder_gpu=False,     # Use CPU for embedder (set True if GPU)
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None,
            )
            logger.info(
                f"DeepSORT initialized | max_age={max_age}, "
                f"n_init={min_hits}, iou={iou_thresh}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DeepSORT: {e}", exc_info=True)
            return False

    def _detections_to_deepsort_format(
        self, detections: List[Detection]
    ) -> List[Tuple]:
        """
        Convert our Detection objects to the format DeepSORT expects.
        
        DeepSORT input format:
          List of tuples: (bbox_xywh, confidence, class_id)
          
          bbox_xywh = [x_left, y_top, width, height]
          Note: DeepSORT wants TOP-LEFT + WIDTH/HEIGHT, not x1y1x2y2!
        
        WHY THIS CONVERSION:
        Our YOLOv8 Detection stores bboxes as (x1,y1,x2,y2) — corner format.
        DeepSORT wants (x,y,w,h) — top-left + size format.
        
        Conversion:
          x = x1
          y = y1
          w = x2 - x1
          h = y2 - y1
        """
        deepsort_input = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            w = x2 - x1
            h = y2 - y1

            # Skip degenerate boxes (zero or negative size)
            if w <= 0 or h <= 0:
                continue

            bbox_xywh = [x1, y1, w, h]
            deepsort_input.append((bbox_xywh, det.confidence, det.class_id))

        return deepsort_input

    def update(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
    ) -> Dict[int, TrackedObject]:
        """
        Update tracker with new frame and detections.
        
        This is the core tracking function called every frame.
        
        WHAT HAPPENS INSIDE update_tracks():
          1. Kalman Filter predicts new positions for all existing tracks
          2. Hungarian Algorithm matches predictions to new detections
          3. Matched tracks are updated with real detection positions
          4. Unmatched tracks continue on Kalman prediction only
          5. Unmatched detections become new tentative tracks
          6. Tracks exceeding max_age without matches are deleted
        
        Args:
            frame:      BGR numpy array (needed for appearance embedding extraction)
            detections: YOLOv8 detections for this frame
            frame_idx:  Current frame index
        
        Returns:
            Dict mapping track_id → TrackedObject for ALL active tracks
        """
        if self._tracker is None:
            logger.error("Tracker not initialized. Call initialize() first.")
            return {}

        self._frame_count += 1
        max_trail=self.cfg.max_trajectory_len

        # Convert detections to DeepSORT format
        ds_input = self._detections_to_deepsort_format(detections)

        # Run DeepSORT update
        # Returns list of Track objects
        # Each Track has:
        #   .track_id       → unique integer ID
        #   .to_ltrb()      → bounding box as [x1,y1,x2,y2] (left-top-right-bottom)
        #   .det_class      → detected class ID
        #   .det_conf       → detection confidence
        #   .is_confirmed() → True if track has been confirmed (seen n_init times)
        try:
            raw_tracks = self._tracker.update_tracks(ds_input, frame=frame)
        except Exception as e:
            logger.warning(f"DeepSORT update failed on frame {frame_idx}: {e}")
            return self._tracks

        # Process returned tracks
        active_ids = set()

        for track in raw_tracks:
            # Only process confirmed tracks (ignore tentative/new ones)
            # Confirmed = seen in at least n_init consecutive frames
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            active_ids.add(track_id)

            # Get bounding box in x1,y1,x2,y2 format
            try:
                ltrb = track.to_ltrb()   # [left, top, right, bottom]
                bbox = (float(ltrb[0]), float(ltrb[1]),
                        float(ltrb[2]), float(ltrb[3]))
            except Exception:
                continue

            # Get class info
            class_id = int(track.det_class) if track.det_class is not None else -1

            class_names = getattr(DETECTION_CONFIG, "class_names", {})
            class_name = class_names.get(class_id, f"class_{class_id}")

            conf = float(track.det_conf) if track.det_conf is not None else 0.0

            # Create or update TrackedObject in our registry
            if track_id not in self._tracks:
                tracked_obj           = TrackedObject(track_id, class_id, class_name, max_trail)
                tracked_obj.birth_frame = frame_idx
                self._tracks[track_id] = tracked_obj
                self._total_tracks_created += 1
                logger.debug(f"New track: #{track_id} ({class_name})")
            else:
                tracked_obj = self._tracks[track_id]

            # Update with latest data
            tracked_obj.update(frame_idx, bbox, conf)

            # Buffer for CSV export
            if self.cfg.save_trajectories:
                self._csv_rows.append(tracked_obj.to_dict(frame_idx))

        # Log stats every 100 frames
        if self._frame_count % 100 == 0:
            logger.info(
                f"Frame {frame_idx:05d} | Active tracks: {len(active_ids)} | "
                f"Total created: {self._total_tracks_created}"
            )

        return self._tracks

    def annotate_frame(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw tracking visualization on the frame.
        
        WHAT WE DRAW:
          1. Bounding box (colored by class)
          2. Track ID label with class name
          3. Trajectory trail (polyline of past positions)
          4. Current position dot
          5. FPS and track count overlay
        
        TRAIL DRAWING WITH cv2.polylines:
          We convert the deque of (frame_idx, cx, cy) to a list of (cx, cy) points,
          then draw them as a connected polyline.
          We fade older points by drawing shorter, thinner lines for visual clarity.
        """
        annotated = frame.copy()

        active_tracks = {
            tid: t for tid, t in self._tracks.items()
            if t.bbox_xyxy is not None
        }

        for track_id, track in active_tracks.items():
            if track.bbox_xyxy is None:
                continue

            x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
            color = self._class_colors.get(track.class_id, (0, 255, 0))

            # --- Bounding Box ---
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # --- ID Label ---
            label = f"#{track_id} {track.class_name}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness  = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Background rectangle for label
            label_y1 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + tw + 4, y1),
                color,
                cv2.FILLED,
            )
            brightness  = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color  = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - baseline - 2),
                font, font_scale, text_color, thickness, cv2.LINE_AA,
            )

            # --- Trajectory Trail ---
            # Extract (cx, cy) points from trail deque
            trail_points = [(int(cx), int(cy)) for _, cx, cy in track.trail]

            if len(trail_points) >= 2:
                # Draw trail as a series of line segments
                # We fade opacity by varying thickness along the trail
                n = len(trail_points)
                for i in range(1, n):
                    # Older segments are thinner and darker
                    # Fraction 0.0 = oldest, 1.0 = newest
                    frac      = i / n
                    thickness_trail = max(1, int(frac * 3))
                    
                    # Fade color: interpolate between dark and full color
                    faded_color = tuple(int(c * (0.3 + 0.7 * frac)) for c in color)
                    
                    cv2.line(
                        annotated,
                        trail_points[i - 1],
                        trail_points[i],
                        faded_color,
                        thickness_trail,
                        cv2.LINE_AA,
                    )

            # --- Current Position Dot ---
            if trail_points:
                cx, cy = trail_points[-1]
                cv2.circle(annotated, (cx, cy), 4, color, -1)   # Filled dot

        # --- HUD Overlay ---
        self._draw_hud(annotated, fps, len(active_tracks))

        return annotated

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        track_count: int,
    ) -> None:
        """
        Draw Heads-Up Display (HUD) with system stats.
        Drawn directly on the frame (modifies in-place).
        """
        # Semi-transparent dark background bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 28), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Active Tracks: {track_count}",
                    (160, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Total Tracks Created: {self._total_tracks_created}",
                    (10, 58), font, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Frame: {self._frame_count}",
                    (400, 28), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    def save_trajectories_csv(self, output_path: Path) -> None:
        """
        Save all trajectory data to CSV for use by prediction subsystem.
        
        CSV COLUMNS:
          frame_idx, track_id, class_name, x1, y1, x2, y2,
          cx, cy, vx, vy, confidence, age_frames
        
        This CSV is the INPUT to Day 3 trajectory prediction.
        """
        if not self._csv_rows:
            logger.warning("No trajectory data to save.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._csv_rows[0].keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._csv_rows)

        logger.info(
            f"Saved {len(self._csv_rows)} trajectory records → {output_path}"
        )

    @property
    def tracks(self) -> Dict[int, TrackedObject]:
        return self._tracks

    @property
    def active_tracks(self) -> Dict[int, TrackedObject]:
        """Only tracks with a current bounding box."""
        return {tid: t for tid, t in self._tracks.items() if t.bbox_xyxy is not None}

    def get_stats(self) -> dict:
        return {
            "frames_processed": self._frame_count,
            "total_tracks_created": self._total_tracks_created,
            "currently_active": len(self.active_tracks),
            "csv_rows_buffered": len(self._csv_rows),
        }


# ==============================================================================
# FULL TRACKING PIPELINE RUNNER
# ==============================================================================

def run_tracking(
    sequence_dir: Path = UAVDT_SEQ_DIR,
    display: bool = True,
    max_frames: Optional[int] = None,
    save_video: bool = True,
    save_csv: bool = True,
) -> DeepSORTTracker:
    """
    Run the complete detection + tracking pipeline on the UAVDT image sequence.
    
    PIPELINE:
      Load frames → YOLOv8 detect → DeepSORT track → Annotate → Save
    
    Returns the tracker instance (contains all trajectory data).
    """
    setup_logging(
    level=LOG_CONFIG.level,
    log_to_file=LOG_CONFIG.log_to_file,
    log_file=LOG_CONFIG.system_log,
    )
    logger.info("=" * 60)
    logger.info("DAY 2 — TRACKING PIPELINE STARTING")
    logger.info("=" * 60)

    # --- Initialize detector ---
    detector = YOLODetector()
    if not detector.load_model():
        logger.error("Model load failed. Aborting.")
        return None

    # --- Initialize tracker ---
    tracker = DeepSORTTracker()
    if not tracker.initialize():
        logger.error("Tracker init failed. Aborting.")
        return None

    # --- Load frames ---
    loader = FrameLoader(sequence_dir,assumed_fps=VIDEO_CONFIG.output_fps)
    if len(loader) == 0:
        logger.error("No frames loaded. Aborting.")
        return None

    frame_size = loader.frame_size
    logger.info(f"Loaded {len(loader)} frames, size={frame_size}")

    # --- Setup video writer ---
    output_path = OUTPUT_VIDEO_DIR / "tracking_output.mp4"
    writer = VideoWriter(output_path,VIDEO_CONFIG.output_fps,frame_size)
    writer.open()

    # --- FPS tracking ---
    fps_ema   = 0.0
    fps_alpha = 0.1

    try:
        for frame_idx, frame in loader:
            if max_frames and frame_idx >= max_frames:
                break

            t_start = time.perf_counter()

            # 1. Detect
            detections = detector.predict_frame(frame, frame_idx)

            # 2. Track
            tracks = tracker.update(frame, detections, frame_idx)

            # 3. Annotate
            t_elapsed = time.perf_counter() - t_start
            fps_ema   = fps_alpha * (1.0 / max(t_elapsed, 1e-6)) + (1 - fps_alpha) * fps_ema
            annotated = tracker.annotate_frame(frame, fps_ema)

            # 4. Write
            writer.write(annotated)

            # 5. Display
            if display:
                cv2.imshow("UAV Tracking — Press Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit key pressed.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted.")

    finally:
        writer.release()
        if display:
            cv2.destroyAllWindows()

    # --- Save CSV ---
    if save_csv:
        csv_path = OUTPUT_CSV_DIR / "trajectories.csv"
        tracker.save_trajectories_csv(csv_path)

    # --- Summary ---
    stats = tracker.get_stats()
    logger.info("=" * 60)
    logger.info("TRACKING PIPELINE COMPLETE")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  Output video: {output_path}")
    logger.info("=" * 60)
    return tracker
if __name__ == "__main__":
    run_tracking(display=True)
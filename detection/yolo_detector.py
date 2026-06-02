"""
================================================================================
detection/yolo_detector.py — YOLOv8 Object Detection Pipeline
================================================================================

DAY 1 SUBSYSTEM — DETECTION ONLY (no tracking yet)

WHAT THIS MODULE DOES:
  1. Loads the pretrained YOLOv8n model
  2. Reads UAVDT image frames one-by-one (simulating a video stream)
  3. Runs neural network inference on each frame
  4. Filters detections to our target classes (cars, trucks, buses, etc.)
  5. Draws bounding boxes + labels + confidence scores
  6. Displays real-time annotated frames
  7. Calculates and displays FPS
  8. Saves annotated frames as an output video

HOW YOLOv8 INFERENCE WORKS (Simplified):
  Input image (640×640 normalized tensor)
       ↓
  YOLOv8 backbone (CSPDarknet → feature extraction)
       ↓
  Neck (PANet → multi-scale feature fusion)
       ↓
  Detection head → outputs grid of predictions
       ↓
  Each prediction: [x_center, y_center, width, height, confidence, class_probs]
       ↓
  NMS (Non-Maximum Suppression) → removes duplicate boxes
       ↓
  Final detections: [(x1,y1,x2,y2, confidence, class_id), ...]

BOUNDING BOX FORMAT:
  YOLOv8 outputs boxes in xyxy format (top-left + bottom-right corners)
  x1, y1 = top-left corner (pixels)
  x2, y2 = bottom-right corner (pixels)
  These are scaled back to the original image size automatically by ultralytics.

FPS CALCULATION:
  FPS = 1.0 / time_per_frame
  We use exponential moving average (EMA) to smooth the FPS display:
  ema_fps = alpha * current_fps + (1 - alpha) * previous_ema_fps
  This prevents the displayed FPS from jumping around every frame.

PERFORMANCE BOTTLENECKS TO WATCH:
  - cv2.imshow() adds display latency (~5-10ms per frame)
  - Large images slow down YOLOv8 (imgsz=640 is the sweet spot)
  - CPU inference is 5-20x slower than GPU
  - Disk I/O for reading frames from slow HDDs
================================================================================
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ultralytics provides the YOLOv8 API
# YOLO class handles: model loading, preprocessing, inference, postprocessing
from ultralytics import YOLO

# Our custom modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    YOLO_MODEL,
    UAVDT_SEQ_DIR,
    OUTPUT_VIDEO_DIR,
    DETECTION_CONFIG,
    VIDEO_CONFIG,
    LOG_CONFIG,
)
from utils.logger import get_logger, setup_logging
from utils.frame_loader import FrameLoader
from utils.video_writer import VideoWriter

logger = get_logger(__name__)


# ==============================================================================
# DETECTION DATA STRUCTURE
# ==============================================================================

class Detection:
    """
    Represents a single object detection in one frame.
    
    WHY A CLASS INSTEAD OF A DICT:
    Dicts like {"bbox": ..., "conf": ...} have no type hints, no validation,
    and are error-prone (typos in keys cause KeyErrors at runtime, not compile time).
    A class gives us attribute access, type hints, and __repr__ for debugging.
    """

    def __init__(
        self,
        frame_idx: int,
        bbox_xyxy: Tuple[float, float, float, float],  # x1, y1, x2, y2
        confidence: float,
        class_id: int,
        class_name: str,
    ):
        self.frame_idx  = frame_idx
        self.bbox_xyxy  = bbox_xyxy       # (x1, y1, x2, y2) in pixels
        self.confidence = confidence
        self.class_id   = class_id
        self.class_name = class_name

    @property
    def x1(self) -> float: return self.bbox_xyxy[0]
    @property
    def y1(self) -> float: return self.bbox_xyxy[1]
    @property
    def x2(self) -> float: return self.bbox_xyxy[2]
    @property
    def y2(self) -> float: return self.bbox_xyxy[3]

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/CSV export."""
        return {
            "frame_idx":  self.frame_idx,
            "x1":         round(self.x1, 2),
            "y1":         round(self.y1, 2),
            "x2":         round(self.x2, 2),
            "y2":         round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "class_id":   self.class_id,
            "class_name": self.class_name,
        }

    def __repr__(self) -> str:
        return (
            f"Detection(frame={self.frame_idx}, class={self.class_name}, "
            f"conf={self.confidence:.2f}, "
            f"box=({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}))"
        )


# ==============================================================================
# YOLO DETECTOR CLASS
# ==============================================================================

class YOLODetector:
    """
    Production-style YOLOv8 detection pipeline for image sequences.
    
    Responsibilities:
      - Model loading and validation
      - Frame-by-frame inference
      - Detection filtering (class + confidence)
      - Frame annotation (boxes, labels)
      - FPS tracking
      - Output video saving
    
    Separation of concerns:
      - This class ONLY does detection + annotation
      - Tracking, prediction, collision are handled by separate modules
    """

    def __init__(
        self,
        model_path: Path = YOLO_MODEL,
        config: dict = None,
    ):
        """
        Args:
            model_path: Path to YOLOv8 .pt weights file
            config:     Detection config dict (defaults to DETECTION_CONFIG)
        """
        self.model_path = Path(model_path)
        self.cfg        = config or DETECTION_CONFIG
        self.model: Optional[YOLO] = None

        # FPS tracking — exponential moving average for smooth display
        self._fps_ema         = 0.0
        self._fps_alpha       = 0.1      # EMA smoothing factor (lower = smoother)
        self._total_frames    = 0
        self._total_inference = 0.0      # cumulative inference time (seconds)

        # Statistics
        self._detection_counts: List[int] = []

    def load_model(self) -> bool:
        """
        Load the YOLOv8 model from disk.
        
        HOW YOLO() LOADING WORKS:
          - ultralytics downloads config if needed
          - Loads weights into PyTorch
          - Moves to specified device (CPU/CUDA)
          - Sets model to eval() mode (disables dropout/batchnorm training behavior)
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.model_path.exists():
            logger.error(
                f"Model file not found: {self.model_path}\n"
                f"Run: from ultralytics import YOLO; YOLO('yolov8n.pt') to download"
            )
            return False

        logger.info(f"Loading YOLOv8 model from: {self.model_path}")

        try:
            # YOLO() from ultralytics handles everything:
            # - Reading .pt file
            # - Rebuilding PyTorch model architecture
            # - Loading pretrained weights
            # - Device placement
            self.model = YOLO(str(self.model_path))

            # Move model to specified device (CPU or CUDA GPU)
            # This is done implicitly by ultralytics on first inference
            # but we can force it here
            device = self.cfg.device
            logger.info(f"Model will run on device: {device}")

            # Quick validation — run a dummy inference on a black frame
            # This warms up the model (first inference is always slower due to
            # JIT compilation, memory allocation, etc.)
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy,
                verbose=False,
                device=device,
                imgsz=64,
            )
            logger.info("Model loaded and warmed up successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return False

    def predict_frame(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        """
        Run YOLOv8 inference on a single frame and return filtered detections.
        
        INFERENCE PIPELINE:
          1. Preprocess: resize → normalize → tensor → batch
          2. Forward pass: backbone → neck → head
          3. Postprocess: decode predictions → NMS → filter
        
        WHAT results[0] CONTAINS:
          results[0].boxes.xyxy   → tensor of (x1,y1,x2,y2) per detection
          results[0].boxes.conf   → confidence scores
          results[0].boxes.cls    → class IDs
          results[0].orig_shape   → original (H, W) before any resizing
        
        Args:
            frame:     BGR numpy array (H, W, 3)
            frame_idx: Current frame index (for logging/metadata)
        
        Returns:
            List of Detection objects for this frame
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        conf_thresh = self.cfg.confidence_threshold
        iou_thresh = self.cfg.iou_threshold
        imgsz = self.cfg.imgsz
        target_classes = self.cfg.target_classes
        class_names = self.cfg.class_names
        device         = self.cfg.device
        class_names    = self.cfg.class_names   

        try:
            # model.predict() is the main inference call
            # It handles the entire preprocess→inference→postprocess pipeline
            # verbose=False suppresses ultralytics' own print output
            results = self.model.predict(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                device=device,
                classes=target_classes,    # Pre-filter to our classes in the model
                verbose=False,
            )

        except Exception as e:
            logger.error(f"Inference failed on frame {frame_idx}: {e}")
            return []

        # results is a list with one element per input image
        # Since we pass one frame, results[0] has our detections
        result = results[0]

        detections: List[Detection] = []

        # result.boxes is None if no detections found
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Extract tensors and move to CPU + convert to numpy
        # .cpu() moves from GPU to CPU memory
        # .numpy() converts PyTorch tensor to NumPy array
        boxes_xyxy  = result.boxes.xyxy.cpu().numpy()    # shape: (N, 4)
        confidences = result.boxes.conf.cpu().numpy()    # shape: (N,)
        class_ids   = result.boxes.cls.cpu().numpy()     # shape: (N,)

        # Build Detection objects
        for i in range(len(boxes_xyxy)):
            cls_id   = int(class_ids[i])
            conf     = float(confidences[i])
            bbox     = tuple(float(v) for v in boxes_xyxy[i])   # (x1,y1,x2,y2)
            cls_name = class_names.get(cls_id, f"class_{cls_id}")

            det = Detection(
                frame_idx=frame_idx,
                bbox_xyxy=bbox,
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
            )
            detections.append(det)

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, confidence scores, and FPS on the frame.
        
        OPENCV DRAWING:
          cv2.rectangle(img, pt1, pt2, color, thickness)
            pt1 = (x1, y1) top-left corner
            pt2 = (x2, y2) bottom-right corner
            color = (B, G, R) tuple — note BGR not RGB!
            thickness = pixels (negative = filled)
          
          cv2.putText(img, text, org, font, scale, color, thickness)
            org = (x, y) bottom-left of text starting point
        
        We also draw a semi-transparent background behind each label
        for better readability — this requires a blend operation.
        
        Args:
            frame:      BGR frame to annotate (modified in-place on a copy)
            detections: List of Detection objects for this frame
            fps:        Current FPS to display in top-left corner
        
        Returns:
            Annotated frame (new copy, original unchanged)
        """
        # Work on a copy — never mutate the original frame
        # This allows the caller to use the original if needed
        annotated = frame.copy()

        class_colors = self.cfg.class_colors

        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            color = class_colors.get(det.class_id, (0, 255, 0))

            # --- Draw bounding box ---
            # Thickness=2 gives a clean visible box without being too thick
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

            # --- Build label string ---
            label = f"{det.class_name} {det.confidence:.2f}"

            # --- Measure text size for background rectangle ---
            # getTextSize returns ((text_width, text_height), baseline)
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness  = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw filled background rectangle for label readability
            # Place it just above the bounding box top edge
            label_y1 = max(y1 - th - baseline - 4, 0)
            label_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + tw + 4, label_y2),
                color,
                thickness=cv2.FILLED,
            )

            # Choose text color based on background brightness
            # For bright backgrounds → black text; dark → white text
            brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            # Draw the label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA,   # Anti-aliased text for better appearance
            )

        # --- FPS counter in top-left ---
        fps_label = f"FPS: {fps:.1f}"
        cv2.putText(
            annotated,
            fps_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),   # Yellow
            2,
            cv2.LINE_AA,
        )

        # --- Detection count ---
        count_label = f"Detected: {len(detections)}"
        cv2.putText(
            annotated,
            count_label,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated

    def _update_fps_ema(self, frame_time_s: float) -> float:
        """
        Update exponential moving average FPS and return the smoothed value.
        
        WHY EMA FOR FPS:
        Raw FPS = 1 / frame_time → wildly jumps due to GC, disk reads, etc.
        EMA smooths this: new_ema = alpha * raw + (1-alpha) * old_ema
        Small alpha = more smoothing (slower to respond to real changes)
        Large alpha = less smoothing (more responsive but noisy)
        0.1 is a good production default.
        """
        if frame_time_s <= 0:
            return self._fps_ema

        raw_fps       = 1.0 / frame_time_s
        self._fps_ema = (
            self._fps_alpha * raw_fps
            + (1 - self._fps_alpha) * self._fps_ema
        )
        return self._fps_ema

    def run(
        self,
        sequence_dir: Path = UAVDT_SEQ_DIR,
        output_video_path: Optional[Path] = None,
        display: bool = True,
        max_frames: Optional[int] = None,
    ) -> List[List[Detection]]:
        """
        Main detection loop. Runs the full detection pipeline on the image sequence.
        
        MAIN LOOP STRUCTURE:
          For each frame:
            1. Record start time
            2. Run YOLOv8 inference → get detections
            3. Annotate frame
            4. Write to output video
            5. Display (optional)
            6. Calculate FPS
            7. Check for quit key
        
        Args:
            sequence_dir:     Path to folder of sequential images
            output_video_path: Where to save annotated video (None = auto)
            display:          Whether to show real-time window (False for servers)
            max_frames:       Limit number of frames (None = all) — useful for testing
        
        Returns:
            List of detection lists, one per frame (for downstream use)
        """
        # --- Setup output path ---
        if output_video_path is None:
            output_video_path = OUTPUT_VIDEO_DIR / VIDEO_CONFIG["detection_output_name"]

        # --- Load frames ---
        logger.info(f"Starting detection pipeline on: {sequence_dir}")
        loader = FrameLoader(sequence_dir, assumed_fps=VIDEO_CONFIG["output_fps"])

        if len(loader) == 0:
            logger.error("No frames found. Aborting.")
            return []

        # Get frame size for video writer
        frame_size = loader.frame_size   # (width, height)
        if frame_size is None:
            logger.error("Cannot determine frame size. Aborting.")
            return []

        logger.info(f"Sequence: {len(loader)} frames, size {frame_size}")

        # --- Initialize video writer ---
        writer = VideoWriter(
            output_video_path,
            fps=VIDEO_CONFIG["output_fps"],
            frame_size=frame_size,
            codec=VIDEO_CONFIG["codec"],
        )
        writer.open()

        # --- Detection loop ---
        all_detections: List[List[Detection]] = []
        frame_skip = self.cfg.frame_skip
        fps_display  = 0.0

        try:
            for frame_idx, frame in loader:

                # Respect max_frames limit
                if max_frames is not None and frame_idx >= max_frames:
                    logger.info(f"Reached max_frames limit ({max_frames}). Stopping.")
                    break

                # Frame skipping: skip intermediate frames to process faster
                # frame_skip=2 means process every 3rd frame
                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    continue

                # --- TIME MEASUREMENT START ---
                t_start = time.perf_counter()

                # --- INFERENCE ---
                detections = self.predict_frame(frame, frame_idx)
                all_detections.append(detections)

                # --- TIME MEASUREMENT END ---
                t_end      = time.perf_counter()
                frame_time = t_end - t_start

                # Update FPS
                fps_display = self._update_fps_ema(frame_time)
                self._total_frames    += 1
                self._total_inference += frame_time

                # --- ANNOTATE ---
                annotated_frame = self.annotate_frame(frame, detections, fps_display)

                # --- WRITE TO VIDEO ---
                writer.write(annotated_frame)

                # --- LOG PROGRESS ---
                if frame_idx % 50 == 0:
                    logger.info(
                        f"Frame {frame_idx:05d}/{len(loader)} | "
                        f"FPS: {fps_display:.1f} | "
                        f"Detections: {len(detections)}"
                    )

                # --- DISPLAY (optional) ---
                if display:
                    cv2.imshow("UAV Detection — Press Q to quit", annotated_frame)
                    
                    # cv2.waitKey(1) waits 1ms for a keypress
                    # Returns -1 if no key pressed, else the ASCII code
                    # ord('q') = 113 — exit on 'q' key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit key pressed. Stopping detection.")
                        break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping.")

        finally:
            # ALWAYS release writer — critical for file integrity
            writer.release()
            if display:
                cv2.destroyAllWindows()

        # --- Summary statistics ---
        avg_fps = (
            self._total_frames / self._total_inference
            if self._total_inference > 0 else 0
        )
        total_dets = sum(len(d) for d in all_detections)

        logger.info("=" * 60)
        logger.info("DETECTION PIPELINE COMPLETE")
        logger.info(f"  Frames processed : {self._total_frames}")
        logger.info(f"  Total detections : {total_dets}")
        logger.info(f"  Average FPS      : {avg_fps:.1f}")
        logger.info(f"  Avg time/frame   : {(self._total_inference/max(self._total_frames,1))*1000:.1f} ms")
        logger.info(f"  Output video     : {output_video_path}")
        logger.info("=" * 60)

        return all_detections


# ==============================================================================
# STANDALONE RUNNER
# ==============================================================================

def run_detection(
    sequence_dir: Path = UAVDT_SEQ_DIR,
    display: bool = True,
    max_frames: Optional[int] = None,
    save_video: bool = True,
) -> List[List[Detection]]:
    """
    Convenience function to run the detection pipeline.
    Can be called from main.py or other modules.
    """
    setup_logging(**{k: v for k, v in LOG_CONFIG.items() if k != "log_file"},
                  log_file=LOG_CONFIG["log_file"])

    detector = YOLODetector()
    if not detector.load_model():
        raise RuntimeError("Failed to load YOLO model")
    if not detector.load_model():
        logger.error("Cannot start detection — model load failed.")
        return []

    output_path = OUTPUT_VIDEO_DIR / VIDEO_CONFIG["detection_output_name"] if save_video else None

    return detector.run(
        sequence_dir=sequence_dir,
        output_video_path=output_path,
        display=display,
        max_frames=max_frames,
    )


if __name__ == "__main__":
    # Run detection directly: python detection/yolo_detector.py
    run_detection(display=True)
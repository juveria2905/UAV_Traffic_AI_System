"""
================================================================================
utils/video_writer.py — OpenCV VideoWriter Wrapper
================================================================================

WHY WE NEED A WRAPPER:
OpenCV's VideoWriter is powerful but has footguns:
  - You must specify codec BEFORE opening the file
  - Width/Height must match EVERY frame you write (mismatch = silent corruption)
  - You must call .release() or the file won't be finalized (no data written)
  - Codec availability varies by OS (mp4v works everywhere; H264 needs extras)

This wrapper handles all of that safely and adds:
  - Context manager support (with statement auto-releases)
  - Frame count tracking
  - Resolution validation
  - Automatic directory creation

HOW cv2.VideoWriter WORKS INTERNALLY:
  1. You specify: filepath, fourcc codec, fps, (width, height)
  2. It opens the file and writes the video container header
  3. Each .write(frame) call encodes + appends one frame
  4. .release() writes the final index/footer and closes the file
  
FOURCC (Four Character Code):
A 4-letter code identifying the video codec:
  - 'mp4v' = MPEG-4 Part 2 (universal, slightly larger files)
  - 'avc1' = H.264 (smaller files, needs extra libs on Linux)
  - 'XVID' = XviD MPEG-4 (good for .avi files)
We use 'mp4v' for maximum compatibility.
================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class VideoWriter:
    """
    Safe wrapper around cv2.VideoWriter with context manager support.
    
    Usage:
        with VideoWriter("outputs/videos/out.mp4", fps=20, frame_size=(1920,1080)) as writer:
            for frame in frames:
                writer.write(frame)
        # File is automatically finalized here
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: Tuple[int, int],   # (width, height) — OpenCV convention
        codec: str = "mp4v",
    ):
        """
        Args:
            output_path: Where to save the video file (.mp4 recommended)
            fps:         Frames per second in the output video
            frame_size:  (width, height) of each frame — MUST be consistent
            codec:       FourCC codec string ('mp4v' is safest)
        """
        self.output_path = Path(output_path)
        self.fps         = fps
        self.frame_size  = frame_size   # (width, height)
        self.codec       = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0
        self._is_open = False

    def open(self) -> bool:
        """
        Initialize the VideoWriter and open the output file.
        
        Returns True if successful, False if failed.
        """
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # cv2.VideoWriter_fourcc converts 4 chars to an integer codec ID
        # Example: cv2.VideoWriter_fourcc('m','p','4','v') → 828601960
        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        # Create the VideoWriter
        # Signature: VideoWriter(filename, fourcc, fps, (width, height))
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.frame_size,    # (width, height) — width first!
        )

        # isOpened() checks if the file was successfully created
        if not self._writer.isOpened():
            logger.error(
                f"Failed to open VideoWriter at: {self.output_path}\n"
                f"  Codec: {self.codec}, FPS: {self.fps}, Size: {self.frame_size}\n"
                f"  Try changing codec to 'XVID' or check write permissions."
            )
            self._is_open = False
            return False

        self._is_open = True
        logger.info(
            f"VideoWriter opened: {self.output_path.name} | "
            f"Codec: {self.codec} | FPS: {self.fps} | Size: {self.frame_size}"
        )
        return True

    def write(self, frame: np.ndarray) -> bool:
        """
        Write a single frame to the video file.
        
        The frame MUST:
          - Be a numpy array of dtype uint8
          - Have shape (H, W, 3) — BGR
          - Have the same (W, H) as frame_size passed to __init__
        
        Args:
            frame: OpenCV BGR image array
        
        Returns:
            True if frame was written, False if VideoWriter isn't open
        """
        if not self._is_open or self._writer is None:
            logger.warning("VideoWriter.write() called but writer is not open.")
            return False

        # Validate frame shape matches expected size
        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning(
                f"Frame has unexpected shape {frame.shape}. "
                f"Expected (H, W, 3) BGR array."
            )
            return False

        h, w = frame.shape[:2]
        expected_w, expected_h = self.frame_size

        if (w, h) != (expected_w, expected_h):
            # Auto-resize instead of crashing — production-friendly behavior
            logger.debug(
                f"Frame size mismatch: got ({w},{h}), expected {self.frame_size}. "
                f"Auto-resizing."
            )
            frame = cv2.resize(frame, self.frame_size)

        # Ensure dtype is uint8 (0–255)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        self._writer.write(frame)
        self._frame_count += 1
        return True

    def release(self) -> None:
        """
        Finalize and close the video file.
        
        CRITICAL: Without calling this, the output file will be empty or
        corrupted because the video container footer hasn't been written.
        The context manager (__exit__) calls this automatically.
        """
        if self._writer is not None and self._is_open:
            self._writer.release()
            self._is_open = False
            logger.info(
                f"VideoWriter closed. Saved {self._frame_count} frames → "
                f"{self.output_path}"
            )

    def __enter__(self):
        """Context manager entry — opens the writer."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — always releases writer even on exception."""
        self.release()
        # Return False so exceptions propagate normally
        return False

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    @property
    def is_open(self) -> bool:
        return self._is_open
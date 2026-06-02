"""
================================================================================
utils/frame_loader.py — Image Sequence Frame Loader
================================================================================

WHY WE NEED THIS:
The UAVDT dataset gives us individual JPEG frames, NOT a video file.
Real-world drone systems often work this way — they capture frames and store
them as images for annotation-friendly access.

We need to simulate a video stream from these images. This module handles:
  1. Discovering all frames in a directory (sorted correctly)
  2. Loading them one by one like a video reader
  3. Providing metadata (total frames, FPS, resolution)

CRITICAL SORTING ISSUE:
File systems don't always return files in alphabetical/numerical order.
M0203_img000010.jpg must come AFTER M0203_img000009.jpg.
We sort by the numeric part embedded in the filename.

IMAGE LOADING WITH OPENCV:
cv2.imread(path) reads an image file into a NumPy array.
Shape: (height, width, channels) where channels = BGR (not RGB!)
Dtype: uint8 (pixel values 0–255)

WHY BGR NOT RGB?
OpenCV was originally designed for C/C++ in the early 2000s.
The developers chose BGR for memory layout reasons. Now it's a historical
quirk — every OpenCV display function expects BGR, but PyTorch/PIL expects RGB.
We handle conversions explicitly where needed.
================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
import re

from utils.logger import get_logger

logger = get_logger(__name__)


class FrameLoader:
    """
    Loads a sequence of image frames from a directory and yields them
    one by one, simulating a video stream.
    
    Supports: .jpg, .jpeg, .png, .bmp
    
    Usage:
        loader = FrameLoader("datasets/uavdt/raw/M0203")
        for frame_idx, frame in loader:
            # frame is a numpy array: (H, W, 3) in BGR
            process(frame)
    """

    # Supported image extensions
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        sequence_dir: str | Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
        assumed_fps: float = 20.0,
    ):
        """
        Args:
            sequence_dir: Path to folder containing numbered image frames
            start_frame:  Index of the first frame to load (0-indexed)
            end_frame:    Index of the last frame (None = load all)
            resize:       Optional (width, height) to resize each frame to
            assumed_fps:  FPS to report for image sequences (no real FPS in images)
        """
        self.sequence_dir = Path(sequence_dir)
        self.start_frame  = start_frame
        self.end_frame    = end_frame
        self.resize       = resize
        self.assumed_fps  = assumed_fps

        # Discover and sort all frame paths
        self.frame_paths: List[Path] = self._discover_frames()

        # Apply start/end slicing
        self.frame_paths = self.frame_paths[start_frame:end_frame]

        logger.info(
            f"FrameLoader initialized: {len(self.frame_paths)} frames "
            f"from '{self.sequence_dir.name}'"
        )

        if len(self.frame_paths) == 0:
            logger.error(
                f"No image frames found in: {self.sequence_dir}\n"
                f"Expected files like: M0203_img000001.jpg"
            )

    def _discover_frames(self) -> List[Path]:
        """
        Find all image files in the directory and sort them numerically.
        
        WHY NUMERICAL SORT:
        String sort: ['img10.jpg', 'img2.jpg', 'img1.jpg'] (WRONG ORDER)
        Numerical:   ['img1.jpg', 'img2.jpg', 'img10.jpg'] (CORRECT)
        
        We extract the number from the filename using regex and sort by it.
        """
        if not self.sequence_dir.exists():
            logger.error(f"Sequence directory does not exist: {self.sequence_dir}")
            return []

        # Collect all files with supported image extensions
        all_files = [
            f for f in self.sequence_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTS
        ]

        if not all_files:
            logger.warning(f"No image files found in: {self.sequence_dir}")
            return []

        def extract_number(path: Path) -> int:
            """Extract the last sequence of digits from a filename."""
            # re.findall returns all digit groups; we take the last one
            # Example: "M0203_img000042.jpg" → ["0203", "000042"] → 42
            numbers = re.findall(r"\d+", path.stem)
            return int(numbers[-1]) if numbers else 0

        # Sort by extracted frame number
        sorted_files = sorted(all_files, key=extract_number)
        logger.debug(
            f"Discovered {len(sorted_files)} frames. "
            f"First: {sorted_files[0].name}, Last: {sorted_files[-1].name}"
        )
        return sorted_files

    def __len__(self) -> int:
        """Total number of frames available."""
        return len(self.frame_paths)

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over frames, yielding (frame_index, frame_array) tuples.
        
        The frame_array is:
          - NumPy array
          - Shape: (H, W, 3) — height × width × BGR channels
          - Dtype: uint8 — pixel values 0–255
        """
        for idx, path in enumerate(self.frame_paths):
            frame = self._load_frame(path)
            if frame is None:
                logger.warning(f"Skipping corrupted/unreadable frame: {path.name}")
                continue
            yield idx, frame

    def _load_frame(self, path: Path) -> Optional[np.ndarray]:
        """
        Load a single image file as a NumPy array (BGR).
        
        cv2.imread returns None if:
          - File doesn't exist
          - File is corrupted
          - Unsupported format
        We check for this and log it.
        """
        frame = cv2.imread(str(path))

        if frame is None:
            logger.error(f"cv2.imread failed for: {path}")
            return None

        # Optionally resize the frame
        # cv2.resize(src, (width, height)) — note: width before height!
        if self.resize is not None:
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)

        return frame

    def get_frame_at(self, index: int) -> Optional[np.ndarray]:
        """
        Load a specific frame by index (random access).
        Useful for testing without iterating the whole sequence.
        """
        if index < 0 or index >= len(self.frame_paths):
            logger.error(f"Frame index {index} out of range [0, {len(self.frame_paths)-1}]")
            return None
        return self._load_frame(self.frame_paths[index])

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        """
        Return (width, height) of the first frame.
        Returns None if no frames are available.
        Note: OpenCV uses (width, height) for size but (height, width) for shape.
        """
        if not self.frame_paths:
            return None
        frame = self._load_frame(self.frame_paths[0])
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return (w, h)

    @property
    def fps(self) -> float:
        """FPS (assumed for image sequences, no real FPS embedded in JPEGs)."""
        return self.assumed_fps

    def summary(self) -> dict:
        """Return a summary dict for logging/display."""
        size = self.frame_size
        return {
            "source_dir": str(self.sequence_dir),
            "total_frames": len(self),
            "frame_size": size,
            "assumed_fps": self.assumed_fps,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }
"""
Shared pytest fixtures for UAV system tests.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np


@pytest.fixture
def sample_tracks():
    return [
        {"id": 1, "bbox": [100, 100, 160, 150], "center": (130, 125),
         "class": "car", "confidence": 0.92, "velocity": (2.0, 1.5)},
        {"id": 2, "bbox": [200, 200, 260, 250], "center": (230, 225),
         "class": "car", "confidence": 0.87, "velocity": (-1.5, 2.0)},
        {"id": 3, "bbox": [400, 100, 460, 150], "center": (430, 125),
         "class": "person", "confidence": 0.75, "velocity": (0.5, 0.5)},
    ]


@pytest.fixture
def sample_predictions():
    return {
        1: [(132, 126.5), (134, 128), (136, 129.5), (138, 131), (140, 132.5)],
        2: [(228.5, 227), (227, 229), (225.5, 231), (224, 233), (222.5, 235)],
        3: [(430.5, 125.5), (431, 126), (431.5, 126.5), (432, 127), (432.5, 127.5)],
    }


@pytest.fixture
def sample_collisions():
    return [
        {
            "track_a": 1, "track_b": 2,
            "ttc": 1.2,
            "distance_px": 45.0,
            "center": (180, 175),
            "bbox": [90, 90, 270, 260],
            "severity": "HIGH",
        }
    ]


@pytest.fixture
def sample_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (20, 25, 20)
    return frame
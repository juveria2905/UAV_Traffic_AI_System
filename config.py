"""
================================================================================
config.py — Unified Configuration System with Attribute Access
================================================================================
"""

from pathlib import Path
from typing import Optional
import os


class ConfigObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                if all(isinstance(k, str) for k in value.keys()):
                    setattr(self, key, ConfigObject(**value))
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return str(self.__dict__)


class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.datasets_dir = self.base_dir / "datasets"
        self.models_dir = self.base_dir / "models"
        self.outputs_dir = self.base_dir / "outputs"

        self.paths = ConfigObject(
            base_dir=self.base_dir,
            datasets_dir=self.datasets_dir,
            models_dir=self.models_dir,
            outputs_dir=self.outputs_dir,
            output_video_dir=self.outputs_dir / "videos",
            output_csv_dir=self.outputs_dir / "csv",
            output_logs_dir=self.outputs_dir / "logs",
            uavdt_seq_dir=self.datasets_dir / "uavdt" / "raw" / "M0203",
            yolo_model_path=self.models_dir / "yolov8n.pt",
            memory_export_json=self.outputs_dir / "csv" / "decision_memory.json",
            reasoning_json=self.outputs_dir / "csv" / "reasoning_chains.json",
        )

        for attr_name in ["output_video_dir", "output_csv_dir", "output_logs_dir"]:
            path = getattr(self.paths, attr_name)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)

        self.detection = ConfigObject(
            confidence_threshold=0.35,
            iou_threshold=0.45,
            imgsz=640,
            device="cpu",
            frame_skip=0,
            target_classes=[0, 2, 3, 5, 7],
            class_names={
                0: "person",
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck",
            },
            class_colors={
                0: (255, 0, 0),
                2: (0, 255, 0),
                3: (0, 165, 255),
                5: (255, 255, 0),
                7: (255, 0, 255),
            }
        )

        self.tracking = ConfigObject(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            max_trajectory_len=50,
            save_trajectories=True,
        )

        self.prediction = ConfigObject(
            prediction_horizon=30,
            predictor_type="linear",
            min_history_len=5,
        )

        self.collision = ConfigObject(
            ttc_high_risk_seconds=3.0,
            ttc_medium_risk_seconds=6.0,
            min_collision_distance_px=200,
            risk_levels={
                "HIGH":   {"color": (0, 0, 255),   "label": "HIGH RISK"},
                "MEDIUM": {"color": (0, 165, 255),  "label": "MEDIUM RISK"},
                "LOW":    {"color": (0, 255, 255),  "label": "LOW RISK"},
            }
        )

        self.agent = ConfigObject(
            enable_confidence_scorer=True,
            enable_reasoning_chains=True,
            enable_decision_memory=True,
            enable_conflict_resolver=True,
            enable_goal_planner=True,
            enable_feedback_engine=True,
            enable_hierarchy_manager=True,
            enable_message_bus=True,

            ttc_emergency_threshold=1.5,
            ttc_reroute_threshold=3.0,
            emergency_confidence_min=0.80,
            reroute_confidence_min=0.55,
            hold_confidence_min=0.50,
            priority_reroute_score=0.70,
            density_hold_threshold=0.60,
            speed_hold_threshold=25.0,
            consecutive_reroute_escalation=3,

            grid_cols=4,
            grid_rows=4,

            feedback_lookback_frames=10,
            decisions_json="agent_decisions.json",
        )

        self.video = ConfigObject(
            output_fps=20.0,
            codec="mp4v",
            detection_output_name="detection_output.mp4",
        )

        self.log = ConfigObject(
            level="INFO",
            log_to_file=True,
            log_to_console=True,
            max_bytes=10 * 1024 * 1024,
            backup_count=5,
            system_log="uav_system.log",
            error_log="uav_errors.log",
            perf_log="uav_performance.log",
        )

        self.api = ConfigObject(
            host="127.0.0.1",  # Changed from 0.0.0.0 (localhost only, connectable)
            port=8000,
        )

        self.dashboard = ConfigObject(
            title="UAV Traffic AI Dashboard",
            page_icon="🚁",
            layout="wide",
            refresh_rate_ms=200,
            max_alert_display=10,
        )

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


cfg = Config()

# Compatibility exports
YOLO_MODEL = cfg.paths.yolo_model_path
UAVDT_SEQ_DIR = cfg.paths.uavdt_seq_dir

OUTPUT_VIDEO_DIR = cfg.paths.output_video_dir
OUTPUT_CSV_DIR   = cfg.paths.output_csv_dir
OUTPUT_LOGS_DIR  = cfg.paths.output_logs_dir

DETECTION_CONFIG  = cfg.detection
TRACKING_CONFIG   = cfg.tracking
PREDICTION_CONFIG = cfg.prediction
COLLISION_CONFIG  = cfg.collision
AGENT_CONFIG      = cfg.agent

VIDEO_CONFIG     = cfg.video
LOG_CONFIG       = cfg.log
API_CONFIG       = cfg.api
DASHBOARD_CONFIG = cfg.dashboard

API_PORT             = cfg.api.port
DASHBOARD_REFRESH_MS = cfg.dashboard.refresh_rate_ms

DECISIONS = ["MONITOR", "HOLD", "REROUTE", "EMERGENCY_STOP"]

DECISION_COLORS_HEX = {
    "MONITOR":       "#00c800",
    "HOLD":          "#ffff00",
    "REROUTE":       "#ffa500",
    "EMERGENCY_STOP": "#ff0000",
}

VIDEO_SOURCE = str(cfg.paths.uavdt_seq_dir)
WRITE_OUTPUT_VIDEO = False
API_HOST = cfg.api.host

DECISION_COLORS_BGR = {
    "MONITOR":       (0, 255, 0),
    "HOLD":          (0, 255, 255),
    "REROUTE":       (0, 165, 255),
    "EMERGENCY_STOP": (0, 0, 255),
}

TTC_THRESHOLD      = cfg.collision.ttc_high_risk_seconds
PRIORITY_WEIGHTS   = {
    "ttc":      0.35,
    "speed":    0.25,
    "density":  0.20,
    "history":  0.20,
}

__all__ = [
    "cfg", "Config",
    "YOLO_MODEL", "UAVDT_SEQ_DIR",
    "OUTPUT_VIDEO_DIR", "OUTPUT_CSV_DIR", "OUTPUT_LOGS_DIR",
    "DETECTION_CONFIG", "TRACKING_CONFIG", "PREDICTION_CONFIG",
    "COLLISION_CONFIG", "AGENT_CONFIG",
    "VIDEO_CONFIG", "LOG_CONFIG", "API_CONFIG", "DASHBOARD_CONFIG",
    "API_PORT", "DASHBOARD_REFRESH_MS",
    "DECISIONS", "DECISION_COLORS_HEX",
    "VIDEO_SOURCE", "WRITE_OUTPUT_VIDEO",
    "API_HOST", "DECISION_COLORS_BGR",
    "TTC_THRESHOLD", "PRIORITY_WEIGHTS",
]
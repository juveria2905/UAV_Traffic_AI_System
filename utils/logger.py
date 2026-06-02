"""
================================================================================
utils/logger.py — Production Logging System
================================================================================

UPGRADE RATIONALE:
  Old: basic StreamHandler + FileHandler, no rotation, no colors, no timing
  New: RotatingFileHandler (10MB × 5 files), separate error.log, colored
       console output, execution timing decorator, startup banner,
       performance metrics logging, ANSI color support with Windows fallback

WHY THIS MATTERS:
  When this project runs for 30 minutes processing 36,000 frames, a single
  unrotated log file would become 500MB+. Rotating handlers keep logs manageable.
  Colored console output lets engineers immediately spot WARNINGS and ERRORS
  at a glance without reading every line.

DEPENDENCIES:
  No external packages — uses only stdlib logging.
  Optional: colorama for Windows ANSI support.
================================================================================
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional, Any


# ==============================================================================
# ANSI COLOR CODES
# ==============================================================================

class _ANSI:
    """ANSI escape codes for terminal colors."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    # Foreground colors
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"


# Windows ANSI initialization
try:
    import colorama
    colorama.init(autoreset=True)
    _ANSI_SUPPORTED = True
except ImportError:
    # Fallback: check if terminal supports ANSI
    _ANSI_SUPPORTED = sys.platform != "win32" or "ANSICON" in os.environ if True else False

import os
_ANSI_SUPPORTED = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ==============================================================================
# COLORED FORMATTER
# ==============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter with per-level ANSI colors.

    Why different colors per level?
      - DEBUG  → grey     (low importance, visual noise)
      - INFO   → cyan     (normal operations)
      - WARNING → yellow  (something to watch)
      - ERROR  → red      (something broke)
      - CRITICAL → bold+red (system failure)

    Color codes are stripped automatically when output is NOT a terminal
    (e.g., when piped to a file or CI/CD system).
    """

    LEVEL_COLORS = {
        logging.DEBUG:    _ANSI.GREY,
        logging.INFO:     _ANSI.CYAN,
        logging.WARNING:  _ANSI.YELLOW,
        logging.ERROR:    _ANSI.RED,
        logging.CRITICAL: _ANSI.BOLD + _ANSI.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format with ANSI colors if supported, plain text otherwise."""
        formatted = super().format(record)

        if not _ANSI_SUPPORTED:
            return formatted

        color = self.LEVEL_COLORS.get(record.levelno, "")
        level_part = f"{color}{record.levelname:<8}{_ANSI.RESET}"

        # Replace the plain levelname with the colored one
        formatted = formatted.replace(record.levelname, level_part, 1)
        return formatted


# ==============================================================================
# PERFORMANCE LOG FILTER
# ==============================================================================

class PerformanceFilter(logging.Filter):
    """
    Only passes log records that contain performance metrics.
    Used to route perf logs to a separate file without polluting system.log.

    We tag perf records by including '[PERF]' in the message.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        return "[PERF]" in record.getMessage()


class ExcludePerformanceFilter(logging.Filter):
    """Excludes perf records from the main system log (inverse of above)."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "[PERF]" not in record.getMessage()


# ==============================================================================
# SETUP FUNCTION
# ==============================================================================

def setup_logging(
    level:          str           = "INFO",
    log_to_file:    bool          = True,
    log_to_console: bool          = True,
    colored:        bool          = True,
    logs_dir:       Optional[Path] = None,
    max_bytes:      int           = 10 * 1024 * 1024,   # 10 MB
    backup_count:   int           = 5,
    system_log:     str           = "uav_system.log",
    error_log:      str           = "uav_errors.log",
    perf_log:       str           = "uav_performance.log",
    fmt:            str           = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
    datefmt:        str           = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure the root logger with rotating file handlers + colored console.

    Call ONCE at application startup (main.py entry point).
    All subsequent get_logger(__name__) calls inherit this configuration.

    Handler hierarchy:
      ConsoleHandler (ColoredFormatter) → stdout
      RotatingFileHandler (system.log)  → all levels (excluding PERF)
      RotatingFileHandler (error.log)   → ERROR + CRITICAL only
      RotatingFileHandler (perf.log)    → PERF-tagged records only

    Args:
        level:        Root log level ("DEBUG" | "INFO" | "WARNING" | "ERROR")
        log_to_file:  Whether to write to rotating log files
        log_to_console: Whether to output to stdout
        colored:      Whether to use ANSI color codes in console output
        logs_dir:     Directory for log files (auto-created if needed)
        max_bytes:    Maximum size of each log file before rotation (bytes)
        backup_count: Number of rotated log files to keep per handler
        system_log:   Filename for main system log
        error_log:    Filename for errors-only log
        perf_log:     Filename for performance metrics log
        fmt:          Log message format string
        datefmt:      Timestamp format string
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Clear existing handlers to prevent duplicates on hot reload
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    plain_formatter  = logging.Formatter(fmt=fmt, datefmt=datefmt)
    colored_fmt      = ColoredFormatter(fmt=fmt, datefmt=datefmt)

    # ── Console Handler ──────────────────────────────────────────────────────
    if log_to_console:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(numeric_level)
        console.setFormatter(colored_fmt if (colored and _ANSI_SUPPORTED) else plain_formatter)
        root.addHandler(console)

    # ── File Handlers ─────────────────────────────────────────────────────────
    if log_to_file and logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Main system log (all messages, no perf)
        sys_handler = logging.handlers.RotatingFileHandler(
            logs_dir / system_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        sys_handler.setLevel(numeric_level)
        sys_handler.setFormatter(plain_formatter)
        sys_handler.addFilter(ExcludePerformanceFilter())
        root.addHandler(sys_handler)

        # Error-only log (easy to check for problems)
        err_handler = logging.handlers.RotatingFileHandler(
            logs_dir / error_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(plain_formatter)
        root.addHandler(err_handler)

        # Performance metrics log
        perf_handler = logging.handlers.RotatingFileHandler(
            logs_dir / perf_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(plain_formatter)
        perf_handler.addFilter(PerformanceFilter())
        root.addHandler(perf_handler)

    # Suppress noisy third-party libraries
    for noisy in ["ultralytics", "torch", "urllib3", "PIL", "matplotlib",
                  "uvicorn.access", "httpx"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Print startup banner to confirm logging is live
    _print_startup_banner()


# ==============================================================================
# STARTUP BANNER
# ==============================================================================

def _print_startup_banner() -> None:
    """
    Print an ASCII art startup banner to console.
    Makes it immediately clear in the terminal that the system has started.
    """
    banner = f"""
{_ANSI.CYAN if _ANSI_SUPPORTED else ''}
╔══════════════════════════════════════════════════════════════════════════╗
║     UAV TRAFFIC AI — Hierarchical Agentic Autonomous Management         ║
║     Logging System Active  |  Production Grade Pipeline                 ║
╚══════════════════════════════════════════════════════════════════════════╝
{_ANSI.RESET if _ANSI_SUPPORTED else ''}"""
    print(banner)


# ==============================================================================
# GET LOGGER
# ==============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage (in every module):
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Subsystem initialized")

    Using __name__ gives: 'detection.yolo_detector', 'prediction.collision_engine'
    This appears in the log format as the module path, making it easy to trace
    which module produced which log line.
    """
    return logging.getLogger(name)


# ==============================================================================
# PERFORMANCE LOGGING HELPERS
# ==============================================================================

def log_performance(
    logger:   logging.Logger,
    module:   str,
    operation: str,
    elapsed_ms: float,
    extra: Optional[dict] = None,
) -> None:
    """
    Log a performance metric in structured format.
    These go to perf.log via the PerformanceFilter.

    Args:
        logger:     The module's logger
        module:     Subsystem name (e.g., "Detection", "Tracking")
        operation:  What was timed (e.g., "inference", "update_tracks")
        elapsed_ms: Time in milliseconds
        extra:      Optional dict of additional metrics (e.g., {"objects": 17})

    Example output:
        [PERF] Detection | inference=31.2ms | objects=17 | fps=32.1
    """
    extra_str = ""
    if extra:
        extra_str = " | " + " | ".join(f"{k}={v}" for k, v in extra.items())

    logger.info(f"[PERF] {module} | {operation}={elapsed_ms:.1f}ms{extra_str}")


# ==============================================================================
# TIMING DECORATOR
# ==============================================================================

def timed(
    module: str = "",
    operation: str = "",
    log_level: str = "debug",
) -> Callable:
    """
    Decorator that measures and logs function execution time.

    Usage:
        @timed(module="Detection", operation="inference")
        def predict_frame(self, frame, frame_idx):
            ...

    On completion, logs:
        [PERF] Detection | inference=31.2ms

    Args:
        module:     Subsystem label for the log
        operation:  Name of the operation being timed
        log_level:  'debug' | 'info' — where to log timing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            t_start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                # Log the exception with traceback before re-raising
                _logger = get_logger(func.__module__ or __name__)
                _logger.error(
                    f"Exception in {func.__qualname__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                raise
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                _logger = get_logger(func.__module__ or __name__)
                _mod = module or func.__module__.split(".")[-1].title()
                _op  = operation or func.__name__

                if log_level == "info":
                    log_performance(_logger, _mod, _op, elapsed_ms)
                else:
                    _logger.debug(f"[PERF] {_mod} | {_op}={elapsed_ms:.1f}ms")

        return wrapper
    return decorator


# ==============================================================================
# MODULE STATISTICS TRACKER
# ==============================================================================

class ModuleStats:
    """
    Tracks running statistics for a module (call count, avg time, error count).
    One instance per subsystem, used for the /api/metrics endpoint.

    Usage:
        stats = ModuleStats("Detection")
        stats.record(elapsed_ms=31.2, success=True)
        print(stats.summary())
    """

    def __init__(self, module_name: str):
        self.module_name  = module_name
        self.call_count   = 0
        self.error_count  = 0
        self.total_ms     = 0.0
        self.min_ms       = float("inf")
        self.max_ms       = 0.0
        self._start_time  = time.time()

    def record(self, elapsed_ms: float, success: bool = True) -> None:
        """Record one execution."""
        self.call_count += 1
        if not success:
            self.error_count += 1
        self.total_ms += elapsed_ms
        self.min_ms    = min(self.min_ms, elapsed_ms)
        self.max_ms    = max(self.max_ms, elapsed_ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_fps(self) -> float:
        return 1000.0 / self.avg_ms if self.avg_ms > 0 else 0.0

    @property
    def uptime_s(self) -> float:
        return time.time() - self._start_time

    def summary(self) -> dict:
        return {
            "module":       self.module_name,
            "calls":        self.call_count,
            "errors":       self.error_count,
            "avg_ms":       round(self.avg_ms, 2),
            "min_ms":       round(self.min_ms, 2) if self.min_ms != float("inf") else 0,
            "max_ms":       round(self.max_ms, 2),
            "avg_fps":      round(self.avg_fps, 1),
            "uptime_s":     round(self.uptime_s, 1),
            "error_rate":   round(self.error_count / max(self.call_count, 1), 4),
        }

    def log_summary(self, logger: logging.Logger) -> None:
        s = self.summary()
        logger.info(
            f"[PERF] {s['module']} stats | "
            f"calls={s['calls']} | "
            f"avg={s['avg_ms']}ms | "
            f"min={s['min_ms']}ms | "
            f"max={s['max_ms']}ms | "
            f"fps={s['avg_fps']} | "
            f"errors={s['errors']}"
        )
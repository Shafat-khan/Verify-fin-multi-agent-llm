"""
verifyfin.utils.logging_utils
================================
Consistent logging configuration for the whole project. Every module
gets a logger via get_logger(__name__); all logs go to both the
console and a persistent file at outputs/logs/verify_fin.log.
"""

from __future__ import annotations

import logging

from verifyfin.utils.io_utils import get_project_root

_LOG_DIR = get_project_root() / "outputs" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "verify_fin.log"

_root_configured = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger for `name` (conventionally __name__ of the calling
    module). Configures the shared 'verifyfin' root logger with console
    + file handlers exactly once, no matter how many modules call this.
    """
    global _root_configured

    if not _root_configured:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(_LOG_FILE)
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger("verifyfin")
        root_logger.setLevel(level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.propagate = False

        _root_configured = True

    return logging.getLogger(name)

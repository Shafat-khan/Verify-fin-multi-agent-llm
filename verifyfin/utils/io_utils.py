"""
verifyfin.utils.io_utils
==========================
Small shared utilities used across the whole project: loading
configs/config.yaml, loading .env, and resolving paths relative to the
project root regardless of which script/directory you run from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from dotenv import load_dotenv

# verify-fin/verifyfin/utils/io_utils.py -> parents[2] == verify-fin/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_env_loaded = False


def get_project_root() -> Path:
    """Absolute path to the verify-fin/ project root."""
    return _PROJECT_ROOT


def load_env() -> None:
    """
    Load variables from the project's .env file into the process
    environment. Safe to call multiple times (idempotent) and does NOT
    override variables already set in the environment (e.g. by a test
    using monkeypatch).
    """
    global _env_loaded
    if not _env_loaded:
        load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
        _env_loaded = True


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """
    Load and return configs/config.yaml (or a custom path) as a dict.
    Also ensures .env has been loaded, since most modules that need
    config also need API keys.
    """
    load_env()
    path = Path(config_path) if config_path else _PROJECT_ROOT / "configs" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Are you running from the project root?"
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """
    Resolve a path relative to the project root (not relative to
    whatever directory a script happens to be run from), creating any
    missing parent directories along the way.
    """
    p = (_PROJECT_ROOT / relative_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

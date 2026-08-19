"""Utilities that make experiments inspectable and repeatable."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch without importing PyTorch for data-only tasks."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata(seed: int, data_manifest: str | Path | None = None) -> dict[str, Any]:
    """Return JSON-serializable metadata to attach to every final run."""
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        metadata["torch"] = None
        metadata["cuda_available"] = False
        metadata["device"] = "unavailable"
    if data_manifest is not None:
        manifest_path = Path(data_manifest)
        metadata["data_manifest"] = str(manifest_path)
        metadata["data_manifest_sha256"] = sha256_file(manifest_path)
    return metadata


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

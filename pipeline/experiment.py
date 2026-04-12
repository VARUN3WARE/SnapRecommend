"""Experiment config and metadata helpers for Phase 2 training."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    LEARNING_RATE,
    NEGATIVE_SAMPLES,
    PHASE_MODE,
    RUNS_DIR,
    SEED,
    TRAIN_SPLIT,
)


@dataclass
class ExperimentConfig:
    run_name: str
    phase_mode: str = PHASE_MODE
    seed: int = SEED
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    train_split: float = TRAIN_SPLIT
    negative_samples: int = NEGATIVE_SAMPLES


@dataclass
class RunMetadata:
    run_id: str
    run_name: str
    phase_mode: str
    seed: int
    started_at: int
    ended_at: int | None = None
    status: str = "running"
    artifacts: dict[str, str] | None = None
    metrics: dict[str, float] | None = None


def create_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{int(time.time())}"


def ensure_artifact_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_experiment_config(config: ExperimentConfig, run_id: str) -> Path:
    ensure_artifact_dirs()
    path = RUNS_DIR / run_id / "config.json"
    _write_json(path, asdict(config))
    return path


def save_run_metadata(metadata: RunMetadata) -> Path:
    ensure_artifact_dirs()
    path = RUNS_DIR / metadata.run_id / "metadata.json"
    _write_json(path, asdict(metadata))
    return path


def load_run_metadata(run_id: str) -> RunMetadata:
    path = RUNS_DIR / run_id / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunMetadata(**payload)


def mark_run_finished(
    run_id: str,
    status: str,
    metrics: dict[str, float] | None = None,
    artifacts: dict[str, str] | None = None,
) -> Path:
    metadata = load_run_metadata(run_id)
    metadata.status = status
    metadata.ended_at = int(time.time())
    metadata.metrics = metrics or {}
    metadata.artifacts = artifacts or {}
    return save_run_metadata(metadata)

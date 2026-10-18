"""Provenance: make every run reproducible, inspectable and resumable.

Three things are recorded for each campaign:

* a **run manifest** -- the exact configuration, a hash of it, the random seed,
  the library versions and the git commit. Two runs with the same config hash
  and seed produce identical results; a different hash tells you immediately
  that something about the setup moved.
* a **checkpoint** after every round -- all observations plus the internal state
  of both random generators, so an interrupted campaign resumes exactly where it
  stopped rather than starting over. When each experiment is expensive, losing a
  campaign to a crashed node is losing real money.
* the **decision and tool-call logs** (written by the campaign itself), so the
  record shows not only what was configured but what was actually done.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_TRACKED_PACKAGES = ("numpy", "scipy", "scikit-learn", "matplotlib", "anthropic")


def canonical_json(obj: Any) -> str:
    """Stable JSON rendering: sorted keys, no whitespace jitter."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict) -> str:
    """Short, stable fingerprint of a configuration."""
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]


def capture_environment() -> dict:
    """Python, platform and library versions -- what the result actually depended on."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return {"python": sys.version.split()[0]}

    packages = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def git_commit() -> str | None:
    """Current commit hash, when the code is running inside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


@dataclass
class RunManifest:
    """Everything needed to understand -- and repeat -- one campaign."""

    config: dict
    config_hash: str = ""
    started_at: str = ""
    finished_at: str | None = None
    environment: dict = field(default_factory=capture_environment)
    git_commit: str | None = field(default_factory=git_commit)

    def __post_init__(self) -> None:
        if not self.config_hash:
            self.config_hash = config_hash(self.config)
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def finish(self) -> "RunManifest":
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @staticmethod
    def load(path: Path) -> "RunManifest":
        data = json.loads(Path(path).read_text())
        return RunManifest(**data)


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def restore_rng(state: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


@dataclass
class Checkpoint:
    """A campaign frozen mid-flight, including both generators' internal state."""

    X_obs: list[list[float]]
    y_obs: list[float]
    history_best: list[float]
    rounds: list[dict]
    rng_state: dict
    oracle_rng_state: dict
    oracle_calls: int
    config_hash: str

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), default=str))
        tmp.replace(path)  # atomic: a crash mid-write cannot corrupt the checkpoint

    @staticmethod
    def load(path: Path) -> "Checkpoint":
        return Checkpoint(**json.loads(Path(path).read_text()))

    @staticmethod
    def exists(path: Path) -> bool:
        return Path(path).is_file()

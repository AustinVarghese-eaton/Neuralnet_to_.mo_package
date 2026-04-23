from __future__ import annotations

from pathlib import Path

def project_root() -> Path:
    """
    Returns the repo root (SurrogateGenerator/).
    Engine/src/surrogate_tool/... -> go up 4 levels to root.
    """
    return Path(__file__).resolve().parents[3]

def runs_root() -> Path:
    return project_root() / "runs"

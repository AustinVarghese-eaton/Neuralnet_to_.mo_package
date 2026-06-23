from __future__ import annotations

from pathlib import Path

def project_root() -> Path:
    """
    Returns the repo root by walking up until we find the unique
    '.surrogate_root' marker file placed at the repo root.
    Falls back to searching for Engine/ + runs/ directories, then
    the original 4-level assumption for dev installs.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".surrogate_root").exists():
            return parent
    for parent in here.parents:
        if (parent / "Engine").is_dir() and (parent / "runs").is_dir():
            return parent
    return here.parents[3]

def runs_root() -> Path:
    return project_root() / "runs"

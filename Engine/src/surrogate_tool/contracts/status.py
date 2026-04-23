from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json


def write_status(path: Path, state: str, message: str, progress: float | None = None) -> None:
    """
    Minimal progress contract.
    Later we will expand states + step timings + warnings.
    """
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": state,
        "message": message,
    }
    if progress is not None:
        payload["progress"] = float(progress)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

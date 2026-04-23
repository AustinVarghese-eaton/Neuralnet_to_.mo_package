from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

from surrogate_tool.paths import runs_root
from surrogate_tool.contracts.status import write_status


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def _safe_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class AttemptInfo:
    attempt_num: int
    attempt_name: str
    attempt_dir: Path
    metrics_path: Path
    report_html: Path
    modelica_pkg_dir: Path


def list_attempts(run_id: str) -> list[AttemptInfo]:
    run_dir = runs_root() / run_id
    attempts_dir = run_dir / "attempts"
    if not attempts_dir.exists():
        return []

    infos: list[AttemptInfo] = []
    for p in sorted(attempts_dir.glob("attempt_*")):
        name = p.name
        try:
            num = int(name.split("_")[1])
        except Exception:
            continue

        metrics_path = p / "reports" / "metrics.json"
        report_html = p / "reports" / "report.html"

        # Modelica dir is nested as: attempt/modelica/<PackageName> (unknown here)
        # We'll store the root modelica folder for this attempt; exporter will create <PackageName> inside it.
        modelica_root = p / "modelica"

        infos.append(
            AttemptInfo(
                attempt_num=num,
                attempt_name=name,
                attempt_dir=p,
                metrics_path=metrics_path,
                report_html=report_html,
                modelica_pkg_dir=modelica_root,
            )
        )
    return infos


def next_attempt_number(run_id: str) -> int:
    infos = list_attempts(run_id)
    if not infos:
        return 1
    return max(i.attempt_num for i in infos) + 1


def score_attempt_rmse_mean_physical(metrics_json: dict) -> Optional[float]:
    """
    Returns rmse_mean in physical units if available, else None.
    """
    try:
        return float(metrics_json["metrics_physical"]["rmse_mean"])
    except Exception:
        return None


def select_best_attempt(run_id: str) -> dict:
    """
    Choose best attempt based on lowest physical RMSE mean.
    Writes runs/<run_id>/latest.json.
    """
    run_dir = runs_root() / run_id
    status_path = run_dir / "status.json"
    write_status(status_path, state="BEST_SELECT", message="Selecting best attempt...", progress=0.9)

    infos = list_attempts(run_id)
    scored: list[tuple[float, AttemptInfo]] = []

    for info in infos:
        m = _safe_read_json(info.metrics_path)
        if not m:
            continue
        s = score_attempt_rmse_mean_physical(m)
        if s is None:
            continue
        scored.append((s, info))

    latest_attempt = f"attempt_{(max([i.attempt_num for i in infos]) if infos else 0):03d}" if infos else None

    if not scored:
        latest = {
            "run_id": run_id,
            "latest_attempt": latest_attempt,
            "best_attempt": None,
            "best_score": None,
            "updated_utc": _utc_now(),
            "note": "No metrics.json found for any attempt; cannot select best.",
        }
        _safe_write_json(run_dir / "latest.json", latest)
        write_status(status_path, state="BEST_SELECT_DONE", message="Best attempt selection complete (none found).", progress=1.0)
        return latest

    scored.sort(key=lambda x: x[0])  # low RMSE is best
    best_score, best_info = scored[0]

    latest = {
        "run_id": run_id,
        "latest_attempt": latest_attempt,
        "best_attempt": best_info.attempt_name,
        "best_score": {"metric": "rmse_mean_physical", "value": best_score},
        "updated_utc": _utc_now(),
        "paths": {
            "best_attempt_dir": str(best_info.attempt_dir),
            "best_metrics": str(best_info.metrics_path),
            "best_report_html": str(best_info.report_html),
            "best_modelica_root": str(best_info.modelica_pkg_dir),
        },
    }

    _safe_write_json(run_dir / "latest.json", latest)
    write_status(status_path, state="BEST_SELECT_DONE", message="Best attempt selected.", progress=1.0)
    return latest


def update_latest_attempt(run_id: str, attempt_num: int) -> None:
    """
    Update runs/<run_id>/latest.json with the latest attempt pointer (best selection done separately).
    """
    run_dir = runs_root() / run_id
    latest_path = run_dir / "latest.json"
    data = _safe_read_json(latest_path) or {"run_id": run_id}
    data["latest_attempt"] = f"attempt_{attempt_num:03d}"
    data["updated_utc"] = _utc_now()
    _safe_write_json(latest_path, data)

from __future__ import annotations

import json

import pytest

from surrogate_tool.attempts.manager import score_attempt_rmse_mean_physical


def test_score_returns_physical_rmse():
    metrics = {"metrics_physical": {"rmse_mean": 3.14}}
    assert score_attempt_rmse_mean_physical(metrics) == pytest.approx(3.14)


def test_score_returns_none_when_key_missing():
    assert score_attempt_rmse_mean_physical({}) is None
    assert score_attempt_rmse_mean_physical({"metrics_physical": {}}) is None


def test_score_returns_none_on_non_numeric():
    assert score_attempt_rmse_mean_physical({"metrics_physical": {"rmse_mean": "n/a"}}) is None


def test_select_best_picks_lowest_rmse(tmp_path, monkeypatch):
    """
    Creates two fake attempt folders with different RMSE values and asserts
    select_best_attempt() picks the one with lower RMSE and writes latest.json.
    """
    import surrogate_tool.attempts.manager as mgr

    run_id = "fake_run"

    # Patch runs_root to return tmp_path
    monkeypatch.setattr(mgr, "runs_root", lambda: tmp_path)

    # Silence write_status by patching it
    monkeypatch.setattr(mgr, "write_status", lambda *a, **kw: None)

    run_dir = tmp_path / run_id
    (run_dir / "attempts" / "attempt_001" / "reports").mkdir(parents=True)
    (run_dir / "attempts" / "attempt_002" / "reports").mkdir(parents=True)

    metrics_001 = {"metrics_physical": {"rmse_mean": 5.0}}
    metrics_002 = {"metrics_physical": {"rmse_mean": 2.0}}

    (run_dir / "attempts" / "attempt_001" / "reports" / "metrics.json").write_text(
        json.dumps(metrics_001), encoding="utf-8"
    )
    (run_dir / "attempts" / "attempt_002" / "reports" / "metrics.json").write_text(
        json.dumps(metrics_002), encoding="utf-8"
    )

    result = mgr.select_best_attempt(run_id)

    assert result["best_attempt"] == "attempt_002"
    assert result["best_score"]["value"] == pytest.approx(2.0)

    latest_path = run_dir / "latest.json"
    assert latest_path.exists()
    saved = json.loads(latest_path.read_text())
    assert saved["best_attempt"] == "attempt_002"

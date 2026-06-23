from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
def test_full_pipeline_roundtrip(tmp_path, monkeypatch):
    """
    200-row synthetic CSV → run_full_attempt() → asserts:
      - mlp_weights.json exists with correct shape keys
      - metrics.json has finite physical RMSE
      - package.mo exists
    DLL compilation is mocked out to avoid gcc dependency.
    """
    import surrogate_tool.paths as paths_mod
    import surrogate_tool.pipeline.fmu_export as fmu_mod

    # ------------------------------------------------------------------
    # 1. Synthetic dataset
    # ------------------------------------------------------------------
    rng = np.random.default_rng(99)
    n = 200
    temp = rng.uniform(20, 120, n)
    voltage = rng.uniform(1.0, 5.0, n)
    power_loss = 0.5 * temp + 2.0 * voltage + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"Temp": temp, "Voltage": voltage, "PowerLoss": power_loss})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # 2. Patch runs_root to use tmp_path
    # ------------------------------------------------------------------
    monkeypatch.setattr(paths_mod, "runs_root", lambda: tmp_path)

    # Patch runs_root in every module that imports it directly
    import surrogate_tool.pipeline.preprocess as pre_mod
    import surrogate_tool.pipeline.eda as eda_mod
    import surrogate_tool.pipeline.split_scale as ss_mod
    import surrogate_tool.pipeline.train as tr_mod
    import surrogate_tool.pipeline.modelica_export as me_mod
    import surrogate_tool.pipeline.report as rep_mod
    import surrogate_tool.attempts.manager as mgr_mod

    for mod in (pre_mod, eda_mod, ss_mod, tr_mod, me_mod, fmu_mod, rep_mod, mgr_mod):
        if hasattr(mod, "runs_root"):
            monkeypatch.setattr(mod, "runs_root", lambda: tmp_path)

    # ------------------------------------------------------------------
    # 3. Mock _compile_dll to skip gcc requirement
    # ------------------------------------------------------------------
    monkeypatch.setattr(fmu_mod, "_compile_dll", lambda *a, **kw: None)

    # ------------------------------------------------------------------
    # 4. Set up run directory
    # ------------------------------------------------------------------
    run_id = "roundtrip_test"
    run_dir = tmp_path / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)

    cfg_dict = {
        "run_id": run_id,
        "dataset_path": str(csv_path),
        "dataset_format": "csv",
        "input_columns": ["Temp", "Voltage"],
        "output_columns": ["PowerLoss"],
        "n_inputs": 2,
        "n_outputs": 1,
        "package_name": "RoundtripModel",
        "random_seed": 42,
        "engine_version": "0.1.0",
    }
    (input_dir / "run_config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"state": "READY", "message": ""}), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 5. Run
    # ------------------------------------------------------------------
    from surrogate_tool.pipeline.orchestrator import run_full_attempt

    result = run_full_attempt(run_id=run_id, attempt_num=1)

    # ------------------------------------------------------------------
    # 6. Assertions
    # ------------------------------------------------------------------
    attempt_dir = run_dir / "attempts" / "attempt_001"

    # mlp_weights.json must exist with correct keys
    weights_path = attempt_dir / "processed" / "mlp_weights.json"
    assert weights_path.exists(), "mlp_weights.json not created"
    weights = json.loads(weights_path.read_text())
    assert "layers" in weights
    assert "input_cols" in weights
    assert "target_cols" in weights
    assert len(weights["layers"]) > 0

    # metrics.json must have finite physical RMSE
    metrics_path = attempt_dir / "reports" / "metrics.json"
    assert metrics_path.exists(), "metrics.json not created"
    metrics = json.loads(metrics_path.read_text())
    rmse = metrics["metrics_physical"]["rmse_mean"]
    assert np.isfinite(rmse), f"RMSE is not finite: {rmse}"

    # package.mo must exist
    pkg_dir = attempt_dir / "modelica" / "RoundtripModel"
    assert (pkg_dir / "package.mo").exists(), "package.mo not created"

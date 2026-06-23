from __future__ import annotations

import json

import numpy as np
import pytest

from surrogate_tool.pipeline.modelica_export import _load_weights_as_modelica_Wb


# ---------------------------------------------------------------------------
# _load_weights_as_modelica_Wb — transpose correctness
# ---------------------------------------------------------------------------

def test_weight_transpose_shape():
    """W stored as Keras (n_in, n_out); after transpose it should be (n_out, n_in)."""
    export = {
        "layers": [
            {"W": [[1.0, 2.0], [3.0, 4.0]], "b": [0.1, 0.2], "activation": "relu"},
        ]
    }
    Ws, bs, acts = _load_weights_as_modelica_Wb(export)
    assert Ws[0].shape == (2, 2)  # (n_out=2, n_in=2)


def test_weight_transpose_values():
    """
    Input W = [[1, 2], [3, 4]] → Keras (n_in=2, n_out=2).
    Transposed for Modelica (n_out=2, n_in=2) → [[1, 3], [2, 4]].
    """
    export = {
        "layers": [
            {"W": [[1.0, 2.0], [3.0, 4.0]], "b": [0.1, 0.2], "activation": "relu"},
        ]
    }
    Ws, bs, acts = _load_weights_as_modelica_Wb(export)
    expected = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_almost_equal(Ws[0], expected)


def test_weight_transpose_rectangular():
    """Non-square layer: W shape (2, 3) Keras → transposed (3, 2) Modelica."""
    export = {
        "layers": [
            {
                "W": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # (n_in=2, n_out=3)
                "b": [0.0, 0.0, 0.0],
                "activation": "relu",
            }
        ]
    }
    Ws, bs, acts = _load_weights_as_modelica_Wb(export)
    assert Ws[0].shape == (3, 2)


def test_activation_mapping():
    export = {
        "layers": [
            {"W": [[1.0]], "b": [0.0], "activation": "relu"},
            {"W": [[1.0]], "b": [0.0], "activation": "linear"},
            {"W": [[1.0]], "b": [0.0], "activation": "RELU"},
        ]
    }
    _, _, acts = _load_weights_as_modelica_Wb(export)
    assert acts[0] == "relu"
    assert acts[1] == "identity"
    assert acts[2] == "relu"


def test_empty_layers_raises():
    with pytest.raises(ValueError):
        _load_weights_as_modelica_Wb({"layers": []})


# ---------------------------------------------------------------------------
# export_modelica — file creation (uses monkeypatch on runs_root)
# ---------------------------------------------------------------------------

def test_export_modelica_creates_files(tmp_path, monkeypatch, minimal_mlp_weights):
    import surrogate_tool.pipeline.modelica_export as me_mod
    import surrogate_tool.contracts.run_config as rc_mod

    run_id = "fake_export_run"
    attempt_num = 1
    pkg_name = "TestModel"

    # Patch runs_root
    monkeypatch.setattr(me_mod, "runs_root", lambda: tmp_path)

    # Silence write_status
    monkeypatch.setattr(me_mod, "write_status", lambda *a, **kw: None)

    # Build run directory structure
    run_dir = tmp_path / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)

    # Create a real CSV so RunConfig validates dataset_path
    csv_path = tmp_path / "data.csv"
    import pandas as pd, numpy as np
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "Temp": rng.uniform(20, 120, 50),
        "Voltage": rng.uniform(1, 5, 50),
        "PowerLoss": rng.uniform(5, 50, 50),
    }).to_csv(csv_path, index=False)

    cfg_dict = {
        "run_id": run_id,
        "dataset_path": str(csv_path),
        "dataset_format": "csv",
        "input_columns": ["Temp", "Voltage"],
        "output_columns": ["PowerLoss"],
        "n_inputs": 2,
        "n_outputs": 1,
        "package_name": pkg_name,
        "random_seed": 42,
        "engine_version": "0.1.0",
    }
    (input_dir / "run_config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")

    # Place mlp_weights.json — use minimal_mlp_weights fixture
    processed_dir = run_dir / "attempts" / "attempt_001" / "processed"
    processed_dir.mkdir(parents=True)
    weights = dict(minimal_mlp_weights)
    weights["input_cols"] = ["Temp", "Voltage"]
    weights["target_cols"] = ["PowerLoss"]
    (processed_dir / "mlp_weights.json").write_text(json.dumps(weights), encoding="utf-8")

    result = me_mod.export_modelica(run_id, attempt_num)

    pkg_dir = run_dir / "attempts" / "attempt_001" / "modelica" / pkg_name
    assert (pkg_dir / "package.mo").exists()
    assert (pkg_dir / "Networks" / "SurrogateMLP.mo").exists()

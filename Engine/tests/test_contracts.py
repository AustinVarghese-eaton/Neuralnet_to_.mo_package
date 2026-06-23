from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from surrogate_tool.contracts.run_config import (
    RunConfig,
    load_run_config,
    save_run_config,
    validate_modelica_name,
)
from surrogate_tool.contracts.status import write_status


# ---------------------------------------------------------------------------
# validate_modelica_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["MyModel", "IGBT_NN", "A", "Model123", "m_1"])
def test_package_name_valid(name):
    assert validate_modelica_name(name) == name


@pytest.mark.parametrize("name", ["1Model", "my model", "", "my-model", "_bad"])
def test_package_name_invalid(name):
    with pytest.raises(ValueError):
        validate_modelica_name(name)


# ---------------------------------------------------------------------------
# load_run_config / save_run_config  — BOM round-trip
# ---------------------------------------------------------------------------

def _make_config_dict(dataset_path: str) -> dict:
    return {
        "run_id": "test_run",
        "dataset_path": dataset_path,
        "dataset_format": "csv",
        "input_columns": ["Temp", "Voltage"],
        "output_columns": ["PowerLoss"],
        "n_inputs": 2,
        "n_outputs": 1,
        "package_name": "TestModel",
        "random_seed": 42,
        "engine_version": "0.1.0",
    }


def test_load_save_roundtrip(tmp_path, synthetic_csv):
    cfg_path = tmp_path / "run_config.json"
    raw = _make_config_dict(str(synthetic_csv))
    save_run_config(RunConfig(**raw), cfg_path)
    loaded = load_run_config(cfg_path)
    assert loaded.run_id == "test_run"
    assert loaded.package_name == "TestModel"
    assert loaded.input_columns == ["Temp", "Voltage"]


def test_load_config_bom(tmp_path, synthetic_csv):
    """Windows tools may write UTF-8 BOM prefix; load_run_config must handle it."""
    cfg_path = tmp_path / "run_config.json"
    raw = _make_config_dict(str(synthetic_csv))
    # Write with BOM manually
    cfg_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(raw).encode("utf-8"))
    loaded = load_run_config(cfg_path)
    assert loaded.run_id == "test_run"


# ---------------------------------------------------------------------------
# write_status
# ---------------------------------------------------------------------------

def test_write_status_required_fields(tmp_path):
    path = tmp_path / "status.json"
    write_status(path, state="RUNNING", message="Training...", progress=0.5)
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["state"] == "RUNNING"
    assert obj["message"] == "Training..."
    assert obj["progress"] == pytest.approx(0.5)
    assert "timestamp_utc" in obj


def test_write_status_no_progress(tmp_path):
    path = tmp_path / "status.json"
    write_status(path, state="DONE", message="Complete")
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert "progress" not in obj
    assert obj["state"] == "DONE"

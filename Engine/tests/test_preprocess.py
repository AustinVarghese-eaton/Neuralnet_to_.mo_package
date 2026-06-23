from __future__ import annotations

import pandas as pd
import pytest

from surrogate_tool.pipeline.preprocess import make_attempt_paths, validate_and_clean


def test_make_attempt_paths_structure(tmp_path):
    paths = make_attempt_paths(tmp_path, 1)
    assert paths.attempt_dir == tmp_path / "attempts" / "attempt_001"
    assert paths.processed_dir.exists()
    assert paths.reports_dir.exists()
    assert paths.logs_dir.exists()
    # Paths are set correctly even though the files themselves don't exist yet
    assert paths.cleaned_csv.parent == paths.processed_dir
    assert paths.pipeline_log.parent == paths.logs_dir


def test_make_attempt_paths_numbering(tmp_path):
    p1 = make_attempt_paths(tmp_path, 1)
    p3 = make_attempt_paths(tmp_path, 3)
    assert p1.attempt_dir.name == "attempt_001"
    assert p3.attempt_dir.name == "attempt_003"


def test_validate_and_clean_removes_duplicates():
    df = pd.DataFrame({
        "Temp": [25.0, 25.0, 50.0],
        "Voltage": [3.0, 3.0, 4.0],
        "PowerLoss": [10.0, 10.0, 20.0],
    })
    cleaned, report = validate_and_clean(df, ["Temp", "Voltage"], ["PowerLoss"])
    assert len(cleaned) < len(df)
    assert len(cleaned) == 2


def test_validate_and_clean_removes_nan():
    df = pd.DataFrame({
        "Temp": [25.0, None, 50.0],
        "Voltage": [3.0, 3.0, 4.0],
        "PowerLoss": [10.0, 15.0, 20.0],
    })
    cleaned, report = validate_and_clean(df, ["Temp", "Voltage"], ["PowerLoss"])
    assert cleaned["Temp"].isna().sum() == 0
    assert len(cleaned) == 2


def test_validate_and_clean_coerces_numeric():
    """Non-numeric values become NaN and are dropped."""
    df = pd.DataFrame({
        "Temp": ["25.0", "bad_value", "50.0"],
        "Voltage": ["3.0", "3.0", "4.0"],
        "PowerLoss": ["10.0", "15.0", "20.0"],
    })
    cleaned, report = validate_and_clean(df, ["Temp", "Voltage"], ["PowerLoss"])
    # 'bad_value' row should be dropped
    assert len(cleaned) == 2


def test_validate_and_clean_report_keys():
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    _, report = validate_and_clean(df, ["A"], ["B"])
    for key in ("schema", "quality", "cleaning", "stats", "outliers"):
        assert key in report

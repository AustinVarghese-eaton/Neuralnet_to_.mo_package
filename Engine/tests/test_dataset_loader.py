from __future__ import annotations

import pytest

from surrogate_tool.io.dataset_loader import infer_format, load_dataset, preview_headers


def test_infer_format_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.touch()
    assert infer_format(f) == "csv"


def test_infer_format_xlsx(tmp_path):
    f = tmp_path / "data.xlsx"
    f.touch()
    assert infer_format(f) == "xlsx"


def test_infer_format_unsupported(tmp_path):
    f = tmp_path / "data.txt"
    f.touch()
    with pytest.raises(ValueError):
        infer_format(f)


def test_preview_headers_csv(synthetic_csv):
    headers = preview_headers(synthetic_csv)
    assert headers == ["Temp", "Voltage", "PowerLoss"]


def test_load_dataset_csv_shape(synthetic_csv):
    df = load_dataset(synthetic_csv)
    assert df.shape == (100, 3)
    assert list(df.columns) == ["Temp", "Voltage", "PowerLoss"]


def test_load_dataset_csv_dtypes(synthetic_csv):
    df = load_dataset(synthetic_csv)
    assert df["Temp"].dtype.kind == "f"
    assert df["PowerLoss"].dtype.kind == "f"

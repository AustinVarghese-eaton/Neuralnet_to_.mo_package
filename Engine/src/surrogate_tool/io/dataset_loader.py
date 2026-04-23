from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def infer_format(dataset_path: Path) -> str:
    ext = dataset_path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".xlsx":
        return "xlsx"
    raise ValueError("Unsupported dataset format. Must be .csv or .xlsx")


def preview_headers(dataset_path: Path, sheet_name: Optional[str] = None) -> list[str]:
    """
    Fast header preview without loading full data.
    CSV: read_csv(nrows=0)
    XLSX: read_excel(nrows=0, sheet_name=...)
    """
    fmt = infer_format(dataset_path)

    if fmt == "csv":
        df0 = pd.read_csv(dataset_path, nrows=0)
        return list(df0.columns)

    # xlsx
    if sheet_name is None:
        # default first sheet
        df0 = pd.read_excel(dataset_path, sheet_name=0, nrows=0, engine="openpyxl")
        return list(df0.columns)

    df0 = pd.read_excel(dataset_path, sheet_name=sheet_name, nrows=0, engine="openpyxl")
    return list(df0.columns)


def load_dataset(dataset_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load full dataset. (Used later by preprocess/EDA/training)
    """
    fmt = infer_format(dataset_path)
    if fmt == "csv":
        return pd.read_csv(dataset_path)

    # xlsx
    if sheet_name is None:
        return pd.read_excel(dataset_path, sheet_name=0, engine="openpyxl")
    return pd.read_excel(dataset_path, sheet_name=sheet_name, engine="openpyxl")

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from surrogate_tool.contracts.run_config import RunConfig, load_run_config, save_run_config
from surrogate_tool.contracts.status import write_status
from surrogate_tool.io.dataset_loader import load_dataset
from surrogate_tool.paths import runs_root


@dataclass(frozen=True)
class PreprocessPaths:
    attempt_dir: Path
    processed_dir: Path
    reports_dir: Path
    logs_dir: Path
    cleaned_csv: Path
    report_json: Path
    pipeline_log: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_attempt_paths(run_dir: Path, attempt_num: int) -> PreprocessPaths:
    attempt_name = f"attempt_{attempt_num:03d}"
    attempt_dir = run_dir / "attempts" / attempt_name
    processed_dir = attempt_dir / "processed"
    reports_dir = attempt_dir / "reports"
    logs_dir = attempt_dir / "logs"
    cleaned_csv = processed_dir / "cleaned_dataset.csv"
    report_json = reports_dir / "preprocess_report.json"
    pipeline_log = logs_dir / "pipeline.log"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return PreprocessPaths(
        attempt_dir=attempt_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
        cleaned_csv=cleaned_csv,
        report_json=report_json,
        pipeline_log=pipeline_log,
    )


def validate_and_clean(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    required = input_cols + output_cols

    report: dict = {
        "timestamp_utc": _utc_now(),
        "schema": {},
        "quality": {},
        "cleaning": {},
        "stats": {},
        "outliers": {},
        "warnings": [],
    }

    # --- schema checks ---
    missing_cols = [c for c in required if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in required]
    report["schema"]["missing_columns"] = missing_cols
    report["schema"]["extra_columns"] = extra_cols
    report["schema"]["all_columns"] = list(df.columns)

    if missing_cols:
        report["warnings"].append(f"Missing required columns: {missing_cols}")

    # --- basic quality ---
    report["quality"]["rows_raw"] = int(df.shape[0])
    report["quality"]["cols_raw"] = int(df.shape[1])
    report["quality"]["duplicate_rows_raw"] = int(df.duplicated().sum())
    report["quality"]["missing_values_by_col_raw"] = {c: int(df[c].isna().sum()) for c in df.columns}

    # --- cleaning step 1: drop duplicates ---
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    report["cleaning"]["dropped_duplicates"] = dup_count

    # --- cleaning step 2: coerce required cols to numeric ---
    coerced = []
    for col in required:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced.append(col)
    report["cleaning"]["coerced_to_numeric"] = coerced

    # --- cleaning step 3: drop rows with missing required fields ---
    before = len(df)
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required).reset_index(drop=True)
    report["cleaning"]["dropped_missing_required"] = int(before - len(df))

    # --- post-clean quality ---
    report["quality"]["rows_clean"] = int(df.shape[0])
    report["quality"]["cols_clean"] = int(df.shape[1])
    report["quality"]["missing_values_by_col_clean"] = {c: int(df[c].isna().sum()) for c in df.columns}

    # --- numeric stats for required columns (min/max/mean/std) ---
    stats = {}
    for c in existing_required:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            stats[c] = {
                "min": float(np.nanmin(s.values)) if len(s) else None,
                "max": float(np.nanmax(s.values)) if len(s) else None,
                "mean": float(np.nanmean(s.values)) if len(s) else None,
                "std": float(np.nanstd(s.values)) if len(s) else None,
            }
    report["stats"]["required_columns"] = stats

    # --- IQR outlier report (does NOT remove) ---
    out = {}
    for c in existing_required:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        out[c] = {
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "lower_bound": float(low),
            "upper_bound": float(high),
            "outlier_count": int(mask.sum()),
            "outlier_fraction": float(mask.mean()) if len(s) else 0.0,
        }
    report["outliers"]["iqr"] = out

    return df, report


def run_preprocess(
    run_id: str,
    attempt_num: int,
    inputs_override: Optional[list[str]] = None,
    outputs_override: Optional[list[str]] = None,
) -> PreprocessPaths:
    """
    Loads runs/<run_id>/input/run_config.json, loads dataset, validates & cleans,
    writes cleaned_dataset.csv and preprocess_report.json in attempt workspace.
    """
    run_dir = runs_root() / run_id
    cfg_path = run_dir / "input" / "run_config.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {cfg_path}")

    cfg = load_run_config(cfg_path)

    # Apply CLI overrides if provided (useful before Excel UI exists)
    if inputs_override is not None:
        cfg.input_columns = inputs_override
        cfg.n_inputs = len(inputs_override)
    if outputs_override is not None:
        cfg.output_columns = outputs_override
        cfg.n_outputs = len(outputs_override)

    if not cfg.input_columns or not cfg.output_columns:
        raise ValueError(
            "RunConfig must include non-empty input_columns and output_columns. "
            "Provide them in run_config.json or use --inputs/--outputs on the CLI."
        )

    save_run_config(cfg, cfg_path)  # keep input contract updated

    # Status update at run root
    status_path = run_dir / "status.json"
    write_status(status_path, state="PREPROCESSING", message="Loading dataset and preprocessing...", progress=0.1)

    paths = make_attempt_paths(run_dir, attempt_num)

    # Load dataset (CSV/XLSX + optional sheet)
    df = load_dataset(Path(cfg.dataset_path), sheet_name=cfg.sheet_name)

    write_status(status_path, state="PREPROCESSING", message="Validating schema and cleaning...", progress=0.4)

    df_clean, report = validate_and_clean(df, cfg.input_columns, cfg.output_columns)

    # Add config echo into report
    report["run_id"] = cfg.run_id
    report["attempt"] = f"attempt_{attempt_num:03d}"
    report["dataset_path"] = cfg.dataset_path
    report["sheet_name"] = cfg.sheet_name
    report["input_columns"] = cfg.input_columns
    report["output_columns"] = cfg.output_columns
    report["package_name"] = cfg.package_name

    # Write outputs
    df_clean.to_csv(paths.cleaned_csv, index=False)
    paths.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Minimal attempt meta
    meta = {
        "run_id": cfg.run_id,
        "attempt_num": attempt_num,
        "attempt_dir": str(paths.attempt_dir),
        "created_utc": _utc_now(),
        "engine_version": cfg.engine_version,
        "artifacts": {
            "cleaned_csv": str(paths.cleaned_csv),
            "preprocess_report": str(paths.report_json),
        },
    }
    (paths.attempt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    write_status(status_path, state="PREPROCESSING_DONE", message="Preprocess complete.", progress=1.0)
    return paths

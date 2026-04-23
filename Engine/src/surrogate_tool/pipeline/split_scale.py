from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from surrogate_tool.contracts.run_config import load_run_config
from surrogate_tool.contracts.status import write_status
from surrogate_tool.paths import runs_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _split_data(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    train: float,
    val: float,
    test: float,
    seed: int,
):
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ValueError("Splits must sum to 1.0 (train+val+test).")

    # train vs temp
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(1 - train), random_state=seed
    )

    # temp -> val and test
    val_ratio_of_temp = val / (val + test)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=(1 - val_ratio_of_temp), random_state=seed
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def _scale_xy(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    Y_train_s = y_scaler.fit_transform(Y_train)
    Y_val_s = y_scaler.transform(Y_val)
    Y_test_s = y_scaler.transform(Y_test)

    return X_train_s, X_val_s, X_test_s, Y_train_s, Y_val_s, Y_test_s, x_scaler, y_scaler


def run_split_scale(run_id: str, attempt_num: int, train=0.70, val=0.15, test=0.15) -> dict:
    """
    Uses cleaned_dataset.csv to generate:
      - raw splits CSVs
      - scaled splits CSVs
      - scalers joblib
      - dataset_splits.npz
      - split_scale_report.json
    """
    run_dir = runs_root() / run_id
    cfg_path = run_dir / "input" / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {cfg_path}")

    cfg = load_run_config(cfg_path)
    if not cfg.input_columns or not cfg.output_columns:
        raise ValueError("run_config.json must contain input_columns and output_columns before split-scale.")

    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_num:03d}"
    processed_dir = attempt_dir / "processed"
    reports_dir = attempt_dir / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "status.json"
    write_status(status_path, state="SPLIT_SCALE", message="Loading cleaned dataset...", progress=0.1)

    cleaned_csv = processed_dir / "cleaned_dataset.csv"
    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"Missing cleaned_dataset.csv at: {cleaned_csv}\n"
            f"Run preprocess first."
        )

    df = pd.read_csv(cleaned_csv)

    # Build X and Y dynamically from config (no hardcoding)
    X = df[cfg.input_columns].copy()
    Y = df[cfg.output_columns].copy()

    write_status(status_path, state="SPLIT_SCALE", message="Splitting train/val/test...", progress=0.35)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = _split_data(
        X, Y, train=train, val=val, test=test, seed=int(cfg.random_seed)
    )

    write_status(status_path, state="SPLIT_SCALE", message="Scaling X/Y using StandardScaler...", progress=0.6)

    X_train_s, X_val_s, X_test_s, Y_train_s, Y_val_s, Y_test_s, x_scaler, y_scaler = _scale_xy(
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    )

    # ---------- Save raw CSV splits ----------
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_val.to_csv(processed_dir / "X_val.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)

    Y_train.to_csv(processed_dir / "Y_train.csv", index=False)
    Y_val.to_csv(processed_dir / "Y_val.csv", index=False)
    Y_test.to_csv(processed_dir / "Y_test.csv", index=False)

    # ---------- Save scaled CSV splits ----------
    pd.DataFrame(X_train_s, columns=cfg.input_columns).to_csv(processed_dir / "X_train_scaled.csv", index=False)
    pd.DataFrame(X_val_s, columns=cfg.input_columns).to_csv(processed_dir / "X_val_scaled.csv", index=False)
    pd.DataFrame(X_test_s, columns=cfg.input_columns).to_csv(processed_dir / "X_test_scaled.csv", index=False)

    pd.DataFrame(Y_train_s, columns=cfg.output_columns).to_csv(processed_dir / "Y_train_scaled.csv", index=False)
    pd.DataFrame(Y_val_s, columns=cfg.output_columns).to_csv(processed_dir / "Y_val_scaled.csv", index=False)
    pd.DataFrame(Y_test_s, columns=cfg.output_columns).to_csv(processed_dir / "Y_test_scaled.csv", index=False)

    # ---------- Save scalers ----------
    joblib.dump(x_scaler, processed_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, processed_dir / "y_scaler.joblib")

    # ---------- Save NPZ bundle ----------
    np.savez(
        processed_dir / "dataset_splits.npz",
        X_train=X_train.values, X_val=X_val.values, X_test=X_test.values,
        Y_train=Y_train.values, Y_val=Y_val.values, Y_test=Y_test.values,
        X_train_scaled=X_train_s, X_val_scaled=X_val_s, X_test_scaled=X_test_s,
        Y_train_scaled=Y_train_s, Y_val_scaled=Y_val_s, Y_test_scaled=Y_test_s,
    )

    report = {
        "timestamp_utc": _utc_now(),
        "run_id": cfg.run_id,
        "attempt": f"attempt_{attempt_num:03d}",
        "dataset_used": str(cleaned_csv),
        "input_columns": cfg.input_columns,
        "output_columns": cfg.output_columns,
        "random_seed": int(cfg.random_seed),
        "splits": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "train_frac": float(len(X_train) / len(X)) if len(X) else 0.0,
            "val_frac": float(len(X_val) / len(X)) if len(X) else 0.0,
            "test_frac": float(len(X_test) / len(X)) if len(X) else 0.0,
        },
        "artifacts": {
            "processed_dir": str(processed_dir),
            "x_scaler": str(processed_dir / "x_scaler.joblib"),
            "y_scaler": str(processed_dir / "y_scaler.joblib"),
            "dataset_splits_npz": str(processed_dir / "dataset_splits.npz"),
        },
    }

    (reports_dir / "split_scale_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_status(status_path, state="SPLIT_SCALE_DONE", message="Split + scale complete.", progress=1.0)

    return {
        "processed_dir": str(processed_dir),
        "report": str(reports_dir / "split_scale_report.json"),
        "npz": str(processed_dir / "dataset_splits.npz"),
        "x_scaler": str(processed_dir / "x_scaler.joblib"),
        "y_scaler": str(processed_dir / "y_scaler.joblib"),
    }

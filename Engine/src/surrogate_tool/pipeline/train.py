from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from surrogate_tool.contracts.run_config import load_run_config
from surrogate_tool.contracts.status import write_status
from surrogate_tool.paths import runs_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _configure_attempt_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"attempt::{log_path}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _rmse_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    return np.sqrt(mse)


def _mae_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)

#model architecture
def _build_mlp(input_dim: int, output_dim: int, hidden: list[int], activation: str = "relu"):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(name="SurrogateMLP")
    model.add(layers.Input(shape=(input_dim,), name="input"))
    for i, h in enumerate(hidden):
        model.add(layers.Dense(h, activation=activation, name=f"dense_{i+1}"))
    model.add(layers.Dense(output_dim, activation="linear", name="output"))
    return model


def run_training(
    run_id: str,
    attempt_num: int,
    hidden: list[int] | None = None,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 500,
    patience: int = 30,
) -> dict:
    run_dir = runs_root() / run_id
    cfg_path = run_dir / "input" / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {cfg_path}")

    cfg = load_run_config(cfg_path)
    if not cfg.input_columns or not cfg.output_columns:
        raise ValueError("run_config.json must contain input_columns and output_columns before training.")

    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_num:03d}"
    processed_dir = attempt_dir / "processed"
    reports_dir = attempt_dir / "reports"
    logs_dir = attempt_dir / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    attempt_logger = _configure_attempt_logger(logs_dir / "pipeline.log")
    status_path = run_dir / "status.json"

    # Required artifacts from split-scale
    X_train_p = processed_dir / "X_train_scaled.csv"
    Y_train_p = processed_dir / "Y_train_scaled.csv"
    X_val_p = processed_dir / "X_val_scaled.csv"
    Y_val_p = processed_dir / "Y_val_scaled.csv"
    X_test_p = processed_dir / "X_test_scaled.csv"
    Y_test_p = processed_dir / "Y_test_scaled.csv"
    x_scaler_p = processed_dir / "x_scaler.joblib"
    y_scaler_p = processed_dir / "y_scaler.joblib"

    missing = [p for p in [X_train_p, Y_train_p, X_val_p, Y_val_p, X_test_p, Y_test_p, x_scaler_p, y_scaler_p] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split/scale artifacts. Run split-scale first.\nMissing:\n" +
            "\n".join([f" - {m}" for m in missing])
        )

    write_status(status_path, state="TRAINING", message="Loading scaled splits...", progress=0.1)

    X_train = pd.read_csv(X_train_p).values
    Y_train = pd.read_csv(Y_train_p).values
    X_val = pd.read_csv(X_val_p).values
    Y_val = pd.read_csv(Y_val_p).values
    X_test = pd.read_csv(X_test_p).values
    Y_test = pd.read_csv(Y_test_p).values

    input_dim = int(X_train.shape[1])
    output_dim = int(Y_train.shape[1])

    if hidden is None:
        hidden = [128, 128, 64]

    attempt_logger.info(f"Training start | run_id={run_id} attempt={attempt_num:03d}")
    attempt_logger.info(f"Input dim={input_dim} Output dim={output_dim} Hidden={hidden}")

    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(int(cfg.random_seed))
    np.random.seed(int(cfg.random_seed))

    write_status(status_path, state="TRAINING", message="Building model...", progress=0.2)

    model = _build_mlp(input_dim=input_dim, output_dim=output_dim, hidden=hidden, activation="relu")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(patience), restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
    ]

    write_status(status_path, state="TRAINING", message="Fitting model...", progress=0.35)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=0,
        callbacks=callbacks,
    )

    write_status(status_path, state="TRAINING", message="Evaluating on test set...", progress=0.8)

    Y_pred_test = model.predict(X_test, verbose=0)

    rmse_scaled = _rmse_per_output(Y_test, Y_pred_test)
    mae_scaled = _mae_per_output(Y_test, Y_pred_test)

    y_scaler = joblib.load(y_scaler_p)
    Y_test_phys = y_scaler.inverse_transform(Y_test)
    Y_pred_phys = y_scaler.inverse_transform(Y_pred_test)

    rmse_phys = _rmse_per_output(Y_test_phys, Y_pred_phys)
    mae_phys = _mae_per_output(Y_test_phys, Y_pred_phys)

    hist_df = pd.DataFrame(history.history)
    hist_path = reports_dir / "train_history.csv"
    hist_df.to_csv(hist_path, index=False)

    model_path = processed_dir / "surrogate_model.keras"
    model.save(model_path)

    export = {
        "input_cols": cfg.input_columns,
        "target_cols": cfg.output_columns,
        "layers": [],
        "activation": "relu",
        "x_scaler": {},
        "y_scaler": {},
    }

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            W, b = layer.get_weights()
            export["layers"].append({
                "name": layer.name,
                "W": W.tolist(),
                "b": b.tolist(),
                "activation": layer.activation.__name__,
            })

    x_scaler = joblib.load(x_scaler_p)
    export["x_scaler"] = {"mean": x_scaler.mean_.tolist(), "scale": x_scaler.scale_.tolist()}
    export["y_scaler"] = {"mean": y_scaler.mean_.tolist(), "scale": y_scaler.scale_.tolist()}

    weights_json_path = processed_dir / "mlp_weights.json"
    weights_json_path.write_text(json.dumps(export, indent=2), encoding="utf-8")

    lines = []
    model.summary(print_fn=lambda x: lines.append(x))

    metrics = {
        "timestamp_utc": _utc_now(),
        "run_id": cfg.run_id,
        "attempt": f"attempt_{attempt_num:03d}",
        "package_name": cfg.package_name,
        "input_columns": cfg.input_columns,
        "output_columns": cfg.output_columns,
        "training": {
            "hidden": hidden,
            "lr": float(lr),
            "batch_size": int(batch_size),
            "epochs_requested": int(epochs),
            "epochs_ran": int(len(hist_df)),
            "best_val_loss": float(np.nanmin(hist_df["val_loss"].values)) if "val_loss" in hist_df.columns else None,
        },
        "metrics_scaled": {
            "rmse_per_output": rmse_scaled.tolist(),
            "mae_per_output": mae_scaled.tolist(),
            "rmse_mean": float(np.mean(rmse_scaled)),
            "mae_mean": float(np.mean(mae_scaled)),
        },
        "metrics_physical": {
            "rmse_per_output": rmse_phys.tolist(),
            "mae_per_output": mae_phys.tolist(),
            "rmse_mean": float(np.mean(rmse_phys)),
            "mae_mean": float(np.mean(mae_phys)),
        },
        "artifacts": {
            "model": str(model_path),
            "weights_json": str(weights_json_path),
            "train_history_csv": str(hist_path),
        },
        "model_summary": lines,
    }

    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    attempt_logger.info(f"Saved model: {model_path}")
    attempt_logger.info(f"Saved metrics: {metrics_path}")
    attempt_logger.info(f"Saved weights: {weights_json_path}")

    write_status(status_path, state="TRAINING_DONE", message="Training complete.", progress=1.0)

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "train_history": str(hist_path),
        "weights_json": str(weights_json_path),
    }

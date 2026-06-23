from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (skipped by default)")


@pytest.fixture()
def synthetic_csv(tmp_path):
    """100-row CSV with 2 inputs (Temp, Voltage) and 1 output (PowerLoss)."""
    rng = np.random.default_rng(42)
    n = 100
    temp = rng.uniform(20, 120, n)
    voltage = rng.uniform(1.0, 5.0, n)
    power_loss = 0.5 * temp + 2.0 * voltage + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"Temp": temp, "Voltage": voltage, "PowerLoss": power_loss})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def minimal_mlp_weights():
    """Minimal mlp_weights.json-style dict: 1 hidden layer, 2 inputs, 1 output."""
    return {
        "input_cols": ["Temp", "Voltage"],
        "target_cols": ["PowerLoss"],
        "layers": [
            {
                "W": [[1.0, 2.0], [3.0, 4.0]],   # shape (2, 2) — Keras (n_in, n_out)
                "b": [0.1, 0.2],
                "activation": "relu",
            },
            {
                "W": [[0.5], [0.6]],               # shape (2, 1) — Keras (n_in, n_out)
                "b": [0.0],
                "activation": "linear",
            },
        ],
        "x_scaler": {"mean": [50.0, 3.0], "scale": [30.0, 1.5]},
        "y_scaler": {"mean": [80.0], "scale": [20.0]},
    }

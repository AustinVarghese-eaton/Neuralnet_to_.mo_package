from __future__ import annotations

import numpy as np
import pandas as pd

from surrogate_tool.pipeline.split_scale import _scale_xy, _split_data


def _make_xy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"A": rng.uniform(0, 1, n), "B": rng.uniform(0, 1, n)})
    Y = pd.DataFrame({"C": rng.uniform(0, 1, n)})
    return X, Y


def test_split_ratios():
    X, Y = _make_xy(200)
    X_tr, X_va, X_te, Y_tr, Y_va, Y_te = _split_data(X, Y, 0.70, 0.15, 0.15, seed=42)
    n = len(X)
    assert abs(len(X_tr) - int(0.70 * n)) <= 1
    assert abs(len(X_va) - int(0.15 * n)) <= 2  # rounding tolerance
    assert abs(len(X_te) - int(0.15 * n)) <= 2
    assert len(X_tr) + len(X_va) + len(X_te) == n


def test_split_sums_to_one():
    X, Y = _make_xy(100)
    X_tr, X_va, X_te, Y_tr, Y_va, Y_te = _split_data(X, Y, 0.70, 0.15, 0.15, seed=0)
    assert len(X_tr) + len(X_va) + len(X_te) == 100


def test_no_data_leakage():
    """
    If scaler is fit on train only, applying to val should NOT produce mean≈0 on val.
    (A leaking scaler fitted on all data would center all splits to mean≈0.)
    """
    X, Y = _make_xy(300)
    X_tr, X_va, X_te, Y_tr, Y_va, Y_te = _split_data(X, Y, 0.70, 0.15, 0.15, seed=42)
    X_tr_s, X_va_s, X_te_s, Y_tr_s, Y_va_s, Y_te_s, x_sc, y_sc = _scale_xy(
        X_tr, X_va, X_te, Y_tr, Y_va, Y_te
    )
    # Train mean should be approximately 0 (scaler centred on train)
    assert abs(X_tr_s[:, 0].mean()) < 0.1
    # Val mean in scaled space should NOT be exactly 0 (different distribution than train)
    # — just confirm the scaler was not fit on val
    # The scaler mean/scale are from train only
    assert x_sc.mean_[0] == pytest.approx(X_tr.iloc[:, 0].mean(), rel=1e-4)


def test_scale_xy_shapes():
    X, Y = _make_xy(200)
    X_tr, X_va, X_te, Y_tr, Y_va, Y_te = _split_data(X, Y, 0.70, 0.15, 0.15, seed=0)
    X_tr_s, X_va_s, X_te_s, Y_tr_s, Y_va_s, Y_te_s, x_sc, y_sc = _scale_xy(
        X_tr, X_va, X_te, Y_tr, Y_va, Y_te
    )
    assert X_tr_s.shape == X_tr.shape
    assert X_va_s.shape == X_va.shape
    assert X_te_s.shape == X_te.shape


import pytest

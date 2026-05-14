"""
Forecast metrics computed at horizons h ∈ {1, 7, 14, 30}.

Inputs are y_true and y_pred of shape (n_windows, H, n_items),
plus mase_denom of shape (n_items,). We accumulate per-item per-h
sums and counts and then aggregate at the end.

NOTE on the headline numbers in the paper.
The per-channel ``MASE = MAE / mase_denom`` column written by this
module is a convenience output; it is *not* what populates paper
Table 1 / 6 / 7. Those tables report the GIFT-Eval-style aggregate
(geometric mean over channels of ``MAE_model / MAE_SeasonalNaive``),
which is recomputed post-hoc from the per-channel MAE column by
``gift_style_mase.py``. Downstream consumers of these CSVs (the
analysis scripts in this repo) likewise only read the MAE rows.
"""
from __future__ import annotations

import numpy as np


HORIZONS = [1, 7, 14, 30]   # 1-indexed: h=1 means first forecast day


def _smape_term(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Standard sMAPE: 200 * |y - y_hat| / (|y| + |y_hat|).

    Convention: when both numerator-relevant values are zero, the term
    is zero (perfect forecast on a zero ground-truth).
    """
    num = 2.0 * np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred)
    out = np.where(den > 0, num / den, 0.0)
    return 100.0 * out


def metrics_at_horizons(
    y_true: np.ndarray, y_pred: np.ndarray, mase_denom: np.ndarray,
    horizons: list[int] = HORIZONS,
) -> dict:
    """Returns a dict keyed by (metric, h) -> per-channel array.

    y_true, y_pred: shape (n_windows, H, n_items), float
    mase_denom:    shape (n_items,)

    For h=k, we average the per-day error over [0, k) and over windows;
    that is, MAE@k = mean over (window, day in [0,k), channel).
    Reporting per-channel: average over (window, day in [0,k)) → array of
    shape (n_items,).
    """
    n_windows, H, n_items = y_true.shape
    out = {}
    abs_err = np.abs(y_true - y_pred)                 # (W, H, C)
    sq_err  = (y_true - y_pred) ** 2                  # (W, H, C)
    smape   = _smape_term(y_true, y_pred)             # (W, H, C)

    for h in horizons:
        if h > H:
            continue
        # Cumulative-mean over the first h forecast days, then average
        # over windows. Result is per-channel: shape (n_items,).
        mae   = abs_err[:, :h, :].mean(axis=(0, 1))
        rmse  = np.sqrt(sq_err[:, :h, :].mean(axis=(0, 1)))
        smap  = smape[:, :h, :].mean(axis=(0, 1))
        mase  = mae / mase_denom

        out[("MAE", h)]   = mae
        out[("RMSE", h)]  = rmse
        out[("SMAPE", h)] = smap
        out[("MASE", h)]  = mase
    return out


def to_long_dataframe(metric_dict: dict, item_ids: list[str],
                      model: str, dataset: str):
    """Flatten the (metric, h) -> (n_items,) dict into long format."""
    import pandas as pd
    rows = []
    for (metric, h), arr in metric_dict.items():
        for i, iid in enumerate(item_ids):
            rows.append({
                "model": model, "dataset": dataset,
                "channel": iid, "metric": metric, "h": h,
                "value": float(arr[i]),
            })
    return pd.DataFrame(rows)

"""
Dataset I/O for §4 zero-shot foundation-model evaluation.

Loads per-channel daily demand from the released CSVs, computes the
chronological 70/15/15 split, and the MASE denominator (lag-7
seasonal-naive in-sample MAE per channel, computed on the train slice;
matches the GIFT-Eval-style definition used in paper §F.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DemandSplit:
    """A single dataset's demand series with its chronological splits."""
    label: str
    item_ids: list[str]
    D: np.ndarray              # shape (T, n_items), float32
    train_end: int             # exclusive
    val_end: int               # exclusive — test = [val_end, T)
    mase_denom: np.ndarray     # shape (n_items,), per-channel MAE of
                               # lag-7 seasonal-naive on the train slice

    @property
    def T(self) -> int:
        return self.D.shape[0]

    @property
    def n_items(self) -> int:
        return self.D.shape[1]

    @property
    def test_start(self) -> int:
        return self.val_end


def load_dataset(out_dir: Path,
                 split_train: float = 0.70,
                 split_val: float = 0.15) -> DemandSplit:
    """Load demand from daily_records.csv into (T, n_items) array."""
    cols_path = out_dir / "demand_signals_cols.txt"
    item_ids = cols_path.read_text().strip().split(",")
    n_items = len(item_ids)
    item_to_col = {iid: j for j, iid in enumerate(item_ids)}

    dr = pd.read_csv(
        out_dir / "daily_records.csv",
        usecols=["day", "item", "demand"],
        dtype={"day": np.int32, "demand": np.int64},
    )
    T = int(dr["day"].max() + 1)
    D = np.zeros((T, n_items), dtype=np.float32)
    days = dr["day"].to_numpy()
    cols = dr["item"].map(item_to_col).to_numpy()
    if pd.isna(cols).any():
        raise ValueError("unknown items in daily_records.csv")
    cols = cols.astype(np.int64)
    D[days, cols] = dr["demand"].to_numpy(dtype=np.float32)

    train_end = int(round(T * split_train))
    val_end = int(round(T * (split_train + split_val)))

    # MASE denominator: per-channel mean absolute lag-7 first difference
    # of the TRAIN slice (seasonal-naive at weekly lag, the GluonTS default
    # for daily frequency and the convention reported in paper §F.1).
    train = D[:train_end]
    SEASONAL_LAG = 7
    diff = np.abs(train[SEASONAL_LAG:] - train[:-SEASONAL_LAG])
    mase_denom = diff.mean(axis=0).astype(np.float32)
    # Guard against zero (constant channel); fall back to 1.0 to avoid div-0
    mase_denom = np.where(mase_denom > 0, mase_denom, 1.0)

    return DemandSplit(
        label=out_dir.name,
        item_ids=item_ids,
        D=D,
        train_end=train_end,
        val_end=val_end,
        mase_denom=mase_denom,
    )


def iter_test_windows(split: DemandSplit, L: int = 512, H: int = 30,
                      stride: int = 30):
    """Yield rolling-origin windows whose forecast horizon lies in test.

    Each window: context indices [t - L, t), forecast indices [t, t + H).
    The first t is split.test_start; the last t satisfies t + H <= T.
    Context is allowed to span train/val/test boundaries (rolling-origin).
    """
    T = split.T
    t = split.test_start
    while t + H <= T:
        if t - L < 0:
            t += stride
            continue
        yield t
        t += stride

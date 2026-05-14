"""Compute GIFT-Eval-style aggregate MASE on the two ISOMORPH baseline
releases for the four foundation models, so the values are directly
comparable in scale to the GIFT-Eval leaderboard.

Definition (matches GluonTS standard MASE + GIFT-Eval aggregation):
  MASE_per_channel(model)   = MAE_model_per_channel  / D_seasonal
  MASE_per_channel(SN)      = MAE_SN_per_channel     / D_seasonal
  RelMASE_per_channel(M)    = MASE_model / MASE_SN  (= MAE_model / MAE_SN)
  Aggregate                 = geometric_mean over channels

  D_seasonal: per-channel mean abs lag-m first difference on the train
              slice, with m=7 (weekly), the GluonTS default for daily
              frequency.

  SN baseline: at test window starting day t with horizon H=30,
              prediction y_pred[t+h] = y[t+h-7] for h in [0, H).

The MAE values for the four models per channel per horizon are read
directly from the per-channel CSVs already produced by the foundation
evaluation runs (no re-inference). Only Seasonal Naive is computed
fresh.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data"
RESULT_DIR = REPO / "results" / "eval" / "baseline_and_scenarios"

L = 512
H = 30
STRIDE = 30
SEASONAL_M = 7

DATASETS = ["output_item50", "output_item200"]
HORIZONS = [1, 7, 14, 30]

MODELS = {
    "Chronos":    "chronos_t5_base_{ds}.csv",
    "Moirai":     "moirai_1_1_R_base_{ds}.csv",
    "TimesFM":    "timesfm_2_0_500m_pytorch_{ds}.csv",
    "Lag-Llama":  "lag_llama_{ds}.csv",
}


def load_demand(ds_dir: Path) -> tuple[np.ndarray, list[str]]:
    item_ids = (ds_dir / "demand_signals_cols.txt").read_text().strip().split(",")
    item_to_col = {iid: j for j, iid in enumerate(item_ids)}
    dr = pd.read_csv(ds_dir / "daily_records.csv",
                     usecols=["day", "item", "demand"],
                     dtype={"day": np.int32, "demand": np.int64})
    T = int(dr["day"].max() + 1)
    D = np.zeros((T, len(item_ids)), dtype=np.float32)
    D[dr["day"].to_numpy(),
      dr["item"].map(item_to_col).to_numpy(dtype=np.int64)] = \
        dr["demand"].to_numpy(dtype=np.float32)
    return D, item_ids


def seasonal_denom(D_train: np.ndarray, m: int) -> np.ndarray:
    """Mean abs lag-m first difference per channel on the train slice."""
    diff = np.abs(D_train[m:] - D_train[:-m])
    den = diff.mean(axis=0).astype(np.float32)
    return np.where(den > 0, den, 1.0)


def seasonal_naive_mae(D: np.ndarray, train_end: int, val_end: int,
                       L: int, H: int, stride: int, m: int) -> np.ndarray:
    """Per-channel-per-horizon MAE of SN(m) under the rolling-origin protocol.

    Returns shape (len(HORIZONS), C). Cumulative-mean over the first h
    forecast days then averaged over windows.
    """
    T, C = D.shape
    test_start = val_end
    starts = list(range(test_start, T - H + 1, stride))
    n_W = len(starts)
    abs_err = np.zeros((n_W, H, C), dtype=np.float32)
    for w, t in enumerate(starts):
        for h in range(H):
            true = D[t + h]
            pred = D[t + h - m]
            abs_err[w, h] = np.abs(true - pred)
    mae_per_h = np.zeros((len(HORIZONS), C), dtype=np.float32)
    for i, h in enumerate(HORIZONS):
        mae_per_h[i] = abs_err[:, :h, :].mean(axis=(0, 1))
    return mae_per_h, n_W


def model_mae(csv_path: Path) -> dict[int, np.ndarray]:
    """Load per-channel MAE at each horizon from a model CSV."""
    df = pd.read_csv(csv_path)
    out = {}
    for h in HORIZONS:
        sub = df[(df["metric"] == "MAE") & (df["h"] == h)]
        sub = sub.sort_values("channel")
        out[h] = sub["value"].to_numpy(dtype=np.float32)
    return out


def gift_aggregate(rel_per_channel: np.ndarray) -> float:
    """Geometric mean across channels."""
    rel_per_channel = np.maximum(rel_per_channel, 1e-12)
    return float(np.exp(np.log(rel_per_channel).mean()))


def main():
    rows = []
    for ds in DATASETS:
        ds_dir = ROOT / ds
        print(f"\n=== {ds} ===")
        D, item_ids = load_demand(ds_dir)
        T, C = D.shape
        train_end = int(round(T * 0.70))
        val_end = int(round(T * 0.85))
        print(f"  T={T}, C={C}, train_end={train_end}, val_end={val_end}")

        d_seasonal = seasonal_denom(D[:train_end], SEASONAL_M)
        print(f"  seasonal-{SEASONAL_M} denom:  "
              f"min={d_seasonal.min():.3f}  "
              f"mean={d_seasonal.mean():.3f}  max={d_seasonal.max():.3f}")

        # Seasonal Naive baseline on rolling windows
        sn_mae, n_W = seasonal_naive_mae(D, train_end, val_end,
                                          L=L, H=H, stride=STRIDE,
                                          m=SEASONAL_M)
        print(f"  windows: {n_W}; SN MAE @ h=30 (mean over channels): "
              f"{sn_mae[3].mean():.3f}")
        sn_mase = sn_mae / d_seasonal[None, :]   # (4, C)

        for model_name, csv_template in MODELS.items():
            csv_name = csv_template.format(ds=ds)
            csv_path = RESULT_DIR / csv_name
            if not csv_path.exists():
                print(f"  [{model_name}] MISSING {csv_name}")
                continue
            mae_dict = model_mae(csv_path)
            for i, h in enumerate(HORIZONS):
                model_mae_arr = mae_dict[h]
                model_mase = model_mae_arr / d_seasonal
                rel = model_mase / sn_mase[i]
                # Cap absurd outliers to avoid inf in geom mean
                rel = np.clip(rel, 1e-3, 1e3)
                gift_mase = gift_aggregate(rel)
                rows.append({
                    "dataset": ds,
                    "model":   model_name,
                    "h":       h,
                    "MAE_mean": float(model_mae_arr.mean()),
                    "MASE_seasonal_mean": float(model_mase.mean()),
                    "RelMASE_geom_over_channels": gift_mase,
                })

    out = pd.DataFrame(rows)
    print("\n\n=== Summary (GIFT-style RelMASE = geom mean over channels of "
          "[MAE_model / MAE_SeasonalNaive(m=7)]) ===")
    pivot = out.pivot_table(index=["dataset", "model"], columns="h",
                            values="RelMASE_geom_over_channels")
    print(pivot.round(3).to_string())

    # Headline aggregate (across horizons): geom mean over h
    print("\n=== Per-(dataset, model) aggregate over horizons "
          "(geom mean over h ∈ {1,7,14,30}) ===")
    agg = (pivot.apply(lambda r: np.exp(np.log(r).mean()), axis=1)
                 .round(3))
    print(agg.to_string())

    out.to_csv(RESULT_DIR / "gift_style_mase.csv", index=False)
    print(f"\nSaved: {RESULT_DIR / 'gift_style_mase.csv'}")


if __name__ == "__main__":
    main()

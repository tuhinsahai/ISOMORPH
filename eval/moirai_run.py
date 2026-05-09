"""
CLI driver: run Moirai zero-shot on one ISOMORPH release.

Outputs (mirror Chronos runner naming):
  results/{model_short}_{dataset}.csv          per-channel long-format metrics
  results/{model_short}_{dataset}_summary.csv  cross-channel mean/median
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_dataset
from metrics import metrics_at_horizons, to_long_dataframe, HORIZONS
from moirai_runner import (
    load_moirai, build_test_data,
    predict_rolling_origin, collect_y_true,
)


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-", "_").replace(".", "_")


def run(out_dir: Path, model_id: str, results_dir: Path,
        L: int, H: int, stride: int,
        num_samples: int, batch_size: int, patch_size,
        max_windows: int | None = None,
        label: str | None = None) -> None:
    print(f"=== {out_dir.name}  with {model_id} ===", file=sys.stderr)
    split = load_dataset(out_dir)
    if label is not None:
        split.label = label
    print(f"  T={split.T}  n_items={split.n_items}  "
          f"train_end={split.train_end}  val_end={split.val_end}  "
          f"test=[{split.test_start}, {split.T})", file=sys.stderr)

    test_data, n_windows = build_test_data(
        split.D, split.item_ids, split.val_end,
        H=H, stride=stride, max_windows=max_windows,
    )
    print(f"  rolling-origin windows: {n_windows}  "
          f"(L={L}, H={H}, stride={stride})", file=sys.stderr)

    predictor = load_moirai(
        model_id,
        prediction_length=H, context_length=L,
        target_dim=split.n_items,
        num_samples=num_samples, patch_size=patch_size,
        batch_size=batch_size,
    )

    t0 = time.time()
    y_pred = predict_rolling_origin(
        predictor, test_data, n_windows, H, split.n_items,
    )
    y_true = collect_y_true(test_data, n_windows, H, split.n_items)
    elapsed = time.time() - t0
    print(f"  inference done in {elapsed/60:.1f} min", file=sys.stderr)

    metric_dict = metrics_at_horizons(y_true, y_pred, split.mase_denom,
                                      horizons=HORIZONS)
    long_df = to_long_dataframe(metric_dict, split.item_ids,
                                model=model_id, dataset=split.label)

    sn = short_name(model_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_long = results_dir / f"{sn}_{split.label}.csv"
    long_df.to_csv(out_long, index=False)
    print(f"  -> {out_long}", file=sys.stderr)

    # Persist raw tensors for post-hoc slicing (e.g. stationary-vs-shock).
    # Moirai's GluonTS test_data hides explicit window starts; reconstruct
    # them from the (val_end, stride, n_windows) triple — the first
    # forecast horizon begins at val_end and successive ones shift by stride.
    starts = [split.val_end + i * stride for i in range(n_windows)]
    out_npz = results_dir / f"{sn}_{split.label}_tensors.npz"
    np.savez_compressed(
        out_npz, y_pred=y_pred, y_true=y_true,
        window_starts=np.asarray(starts, dtype=np.int64),
        item_ids=np.asarray(split.item_ids),
        L=L, H=H, stride=stride, model=model_id, dataset=split.label,
    )
    print(f"  -> {out_npz}", file=sys.stderr)

    summary = (long_df
               .groupby(["model", "dataset", "metric", "h"])["value"]
               .agg(mean="mean", median="median",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75),
                    n="count")
               .reset_index())
    out_sum = results_dir / f"{sn}_{split.label}_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"  -> {out_sum}", file=sys.stderr)

    print("\n  Headline (median across channels):", file=sys.stderr)
    print(summary.pivot_table(index="metric", columns="h",
                              values="median").to_string(),
          file=sys.stderr)


def parse_patch_size(s: str):
    if s == "auto":
        return "auto"
    return int(s)


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(repo / "data"))
    ap.add_argument("--dataset", default="output_item50")
    ap.add_argument("--scenario_path", default=None,
                    help="Path to a scenario directory; "
                         "overrides --root/--dataset when set.")
    ap.add_argument("--label", default=None,
                    help="Output filename label; defaults to out_dir.name.")
    ap.add_argument("--model_id", default="Salesforce/moirai-1.1-R-base")
    ap.add_argument("--out", default=str(
        repo / "results" / "eval" / "baseline_and_scenarios"))
    ap.add_argument("--L", type=int, default=1024,
                    help="context length (Moirai recommends 512-2048)")
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--num_samples", type=int, default=100,
                    help="probabilistic samples; 100 needed for stable median")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="windows per inference batch")
    ap.add_argument("--patch_size", type=parse_patch_size, default=32,
                    help="'auto' or one of {8,16,32,64,128}")
    ap.add_argument("--max_windows", type=int, default=None,
                    help="cap for smoke testing")
    args = ap.parse_args()
    if args.scenario_path is not None:
        out_dir = Path(args.scenario_path)
    else:
        out_dir = Path(args.root) / args.dataset
    run(out_dir,
        args.model_id, Path(args.out),
        L=args.L, H=args.H, stride=args.stride,
        num_samples=args.num_samples,
        batch_size=args.batch_size, patch_size=args.patch_size,
        max_windows=args.max_windows,
        label=args.label)


if __name__ == "__main__":
    main()

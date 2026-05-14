"""
CLI driver: run a foundation model zero-shot on one ISOMORPH release.

Outputs:
  results/{model_short}_{dataset}.csv  long-format per-channel metrics
  results/{model_short}_{dataset}_summary.csv  cross-channel mean/median
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_dataset, iter_test_windows
from metrics import metrics_at_horizons, to_long_dataframe, HORIZONS
from chronos_runner import (
    load_chronos, predict_rolling_origin, collect_y_true,
)


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-", "_")


def run(out_dir: Path, model_id: str, results_dir: Path,
        L: int, H: int, stride: int,
        num_samples: int, channel_batch: int,
        max_windows: int | None = None,
        label: str | None = None) -> None:
    print(f"=== {out_dir.name}  with {model_id} ===", file=sys.stderr)
    split = load_dataset(out_dir)
    if label is not None:
        split.label = label
    print(f"  T={split.T}  n_items={split.n_items}  "
          f"train_end={split.train_end}  val_end={split.val_end}  "
          f"test=[{split.test_start}, {split.T})", file=sys.stderr)

    starts = list(iter_test_windows(split, L=L, H=H, stride=stride))
    if max_windows is not None:
        starts = starts[:max_windows]
    print(f"  rolling-origin windows: {len(starts)}  "
          f"(L={L}, H={H}, stride={stride})", file=sys.stderr)

    pipe = load_chronos(model_id)
    t0 = time.time()
    y_pred = predict_rolling_origin(
        pipe, split.D, starts, L=L, H=H,
        num_samples=num_samples, channel_batch=channel_batch,
    )
    y_true = collect_y_true(split.D, starts, H)
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
    out_npz = results_dir / f"{sn}_{split.label}_tensors.npz"
    np.savez_compressed(
        out_npz, y_pred=y_pred, y_true=y_true,
        window_starts=np.asarray(starts, dtype=np.int64),
        item_ids=np.asarray(split.item_ids),
        L=L, H=H, stride=stride, model=model_id, dataset=split.label,
    )
    print(f"  -> {out_npz}", file=sys.stderr)

    # Cross-channel summary at each (metric, h).
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

    # Print a compact body-table preview.
    print("\n  Headline (median across channels):", file=sys.stderr)
    print(summary.pivot_table(index="metric", columns="h",
                              values="median").to_string(),
          file=sys.stderr)


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(repo / "data"))
    ap.add_argument("--dataset", default="output_item50")
    ap.add_argument("--scenario_path", default=None,
                    help="Path to a scenario directory "
                         "(e.g. data/output_mixture/baseline/seed2025). "
                         "Overrides --root/--dataset when set.")
    ap.add_argument("--label", default=None,
                    help="Output filename label. Defaults to the directory "
                         "name. For scenario paths, set this to the scenario "
                         "name so results from different scenarios don't collide.")
    ap.add_argument("--model_id", default="amazon/chronos-t5-base")
    ap.add_argument("--out", default=str(
        repo / "results" / "eval" / "baseline_and_scenarios"))
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--channel_batch", type=int, default=16)
    ap.add_argument("--max_windows", type=int, default=None,
                    help="cap for smoke testing")
    args = ap.parse_args()
    if args.scenario_path is not None:
        out_dir = Path(args.scenario_path)
    else:
        out_dir = Path(args.root) / args.dataset
    run(out_dir, args.model_id, Path(args.out),
        L=args.L, H=args.H, stride=args.stride,
        num_samples=args.num_samples,
        channel_batch=args.channel_batch,
        max_windows=args.max_windows,
        label=args.label)


if __name__ == "__main__":
    main()

"""Derive edge_list.csv, edge_utilisation.npy, edge_saturation.npy from
a finished scenario directory's shipments.csv + demand_signals.npy +
scenario.json.

The original exp_d_edge_utilisation.py source is lost (only a stale
.pyc survives); this is a clean re-implementation matching the cap
convention used by the existing 13 scenarios (verified by spot-checking
baseline / cap_0.3 / cap_2.5 edge_list.csv values).

Edge capacities follow the simulator's two-tier convention:
  - Upstream edges (15 entries): STATIC_EDGES with num_containers
    multiplied by scenario['containers_scale'] (rounded, min 1), and
    container_volume held fixed.
  - Last-mile edges (PHL->NYC, BAL->NYC): back-solved from the
    realized demand mean using the same formula as the simulator
    (simulate_item50.py).

Daily edge utilisation is computed by streaming shipments.csv and, for
each shipment record, walking its path_nodes/edge_times to attribute
the shipment volume to each (from, to) hop on the day that hop
starts.

Outputs:
  <scenario_dir>/edge_list.csv         per-edge cap_per_day table
  <scenario_dir>/edge_utilisation.npy  shape (T, |E|) float32 in [0, *)
  <scenario_dir>/edge_saturation.npy   shape (T, |E|) uint8 = (util >= tau)

Usage:
    python derive_edge_files.py --scenario_dir <path> [--tau 0.9]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Static upstream graph (matches simulate_item50.py).
# Each entry: (from, to, travel_time_days, container_volume, num_containers).
STATIC_UPSTREAM_EDGES: list[tuple[str, str, int, float, int]] = [
    ("SanFrancisco", "Nashville",     4,  5000.0, 3),
    ("StLouis",      "Nashville",     2,  5000.0, 3),
    ("Orlando",      "Nashville",     2,  5000.0, 3),
    ("Nashville",    "Atlanta",       1, 15000.0, 3),
    ("Atlanta",      "Chicago",       8,  4000.0, 3),
    ("Atlanta",      "Charlotte",     7,  4000.0, 3),
    ("Atlanta",      "Memphis",       7,  4000.0, 3),
    ("Chicago",      "Columbus",      2,  4000.0, 3),
    ("Charlotte",    "Richmond",      2,  4000.0, 3),
    ("Columbus",     "Philadelphia",  2,  4000.0, 3),
    ("Richmond",     "Philadelphia",  1,  4000.0, 3),
    ("Richmond",     "Baltimore",     3,  3000.0, 3),
    ("Columbus",     "Baltimore",     3,  3000.0, 3),
    ("Memphis",      "Baltimore",     2,  3000.0, 3),
]

LAST_MILE_EDGES: list[tuple[str, str, int]] = [
    ("Philadelphia", "NewYork", 1),
    ("Baltimore",    "NewYork", 2),
]

ITEM_AVG_VOL = 2.5
TARGET_RATIO = 1.20
PACKING_EFF = 0.93
PHIL_SHARE = 0.55
BALT_SHARE = 0.45


def back_solve_last_mile_cv(
    n_items: int, demand_signals_path: Path,
) -> tuple[float, float]:
    """Reproduce the simulator's last-mile container_volume back-solve.

    See simulate_item50.py.
    """
    lam = np.load(demand_signals_path)
    actual_mean_lam = float(lam.mean())
    total_demand_vol = n_items * actual_mean_lam * ITEM_AVG_VOL
    raw_needed = total_demand_vol * TARGET_RATIO / PACKING_EFF
    phil_cv = round(raw_needed * PHIL_SHARE / 3 / 100) * 100
    balt_cv = round(raw_needed * BALT_SHARE / 3 / 100) * 100
    return float(phil_cv), float(balt_cv)


def build_edge_df(scenario_dir: Path,
                  scenario: dict,
                  n_items: int = 50) -> pd.DataFrame:
    containers_scale = float(scenario.get("containers_scale", 1.0))
    phil_cv, balt_cv = back_solve_last_mile_cv(
        n_items, scenario_dir / "demand_signals.npy")

    rows = []
    edge_id = 0
    for frm, to, tt, cv, nc in STATIC_UPSTREAM_EDGES:
        nc_scaled = max(1, int(round(nc * containers_scale)))
        rows.append({
            "edge_id": edge_id, "from": frm, "to": to,
            "travel_time_days": tt,
            "container_volume": cv,
            "num_containers": nc_scaled,
            "cap_per_day": cv * nc_scaled,
        })
        edge_id += 1
    for (frm, to, tt), cv in zip(LAST_MILE_EDGES, [phil_cv, balt_cv]):
        nc_scaled = max(1, int(round(3 * containers_scale)))
        rows.append({
            "edge_id": edge_id, "from": frm, "to": to,
            "travel_time_days": tt,
            "container_volume": cv,
            "num_containers": nc_scaled,
            "cap_per_day": cv * nc_scaled,
        })
        edge_id += 1
    return pd.DataFrame(rows)


def compute_utilisation(shipments_path: Path,
                        edge_df: pd.DataFrame,
                        n_days: int,
                        chunksize: int = 500000) -> np.ndarray:
    """Stream shipments.csv and accumulate per-edge daily volume.

    For each shipment row, walks (path_nodes[h], path_nodes[h+1]) and
    attributes its `units` to that edge on the day the hop starts.
    Hop start day = dispatch day + cumulative edge_times of prior hops.
    """
    edge_to_id = {(r["from"], r["to"]): int(r["edge_id"])
                  for _, r in edge_df.iterrows()}
    n_edges = len(edge_df)
    vol = np.zeros((n_days, n_edges), dtype=np.float32)

    rows_seen = 0
    hops_seen = 0
    hops_dropped = 0

    for chunk in pd.read_csv(
        shipments_path,
        usecols=["day", "units", "path_nodes", "edge_times"],
        dtype={"day": np.int32, "units": np.float32},
        chunksize=chunksize,
    ):
        days = chunk["day"].to_numpy()
        units = chunk["units"].to_numpy()
        path_strs = chunk["path_nodes"].to_numpy()
        et_strs = chunk["edge_times"].to_numpy()
        for i in range(len(chunk)):
            path = ast.literal_eval(path_strs[i])
            ets = ast.literal_eval(et_strs[i])
            d0 = int(days[i])
            u = float(units[i])
            cum = 0.0
            for h in range(len(ets)):
                hop = (path[h], path[h + 1])
                eid = edge_to_id.get(hop)
                start_day = d0 + int(round(cum))
                if eid is not None and 0 <= start_day < n_days:
                    vol[start_day, eid] += u
                    hops_seen += 1
                else:
                    hops_dropped += 1
                cum += float(ets[h])
        rows_seen += len(chunk)
        print(f"  processed {rows_seen} rows ({hops_seen} hops, "
              f"{hops_dropped} dropped)", flush=True)

    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario_dir", required=True,
                    help="Path to <output_mixture>/<name>/seed<seed>/")
    ap.add_argument("--n_days", type=int, default=52560)
    ap.add_argument("--n_items", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.9,
                    help="Saturation threshold; matches the convention "
                         "used by the existing 13 scenarios.")
    args = ap.parse_args()

    sdir = Path(args.scenario_dir)
    if not sdir.is_dir():
        sys.exit(f"scenario_dir not found: {sdir}")
    sc = json.loads((sdir / "scenario.json").read_text())

    print(f"=== {sdir.name} (containers_scale="
          f"{sc.get('containers_scale', 1.0)}) ===")

    edge_df = build_edge_df(sdir, sc, n_items=args.n_items)
    edge_df.to_csv(sdir / "edge_list.csv", index=False)
    print(f"  wrote edge_list.csv  (|E|={len(edge_df)})")

    vol = compute_utilisation(
        sdir / "shipments.csv", edge_df, args.n_days)
    cap = edge_df["cap_per_day"].to_numpy(dtype=np.float32)
    util = vol / np.maximum(cap, 1e-9)
    sat = (util >= args.tau).astype(np.uint8)

    np.save(sdir / "edge_utilisation.npy", util.astype(np.float32))
    np.save(sdir / "edge_saturation.npy", sat)
    print(f"  saved edge_utilisation.npy  shape={util.shape}")
    print(f"  saved edge_saturation.npy  shape={sat.shape}  "
          f"saturated_frac={sat.mean():.4f}")


if __name__ == "__main__":
    main()

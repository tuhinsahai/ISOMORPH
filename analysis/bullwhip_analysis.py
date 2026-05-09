"""
Per-node and per-tier bullwhip analysis on the digital-twin sim output.
B_n = Var(inflow_n) / Var(outflow_n)        (Cachon-style amplification ratio)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = os.environ.get("DATA", str(REPO / "data" / "output_item50"))
OUT  = os.environ.get("OUT",  str(REPO / "results" / "bullwhip"))
os.makedirs(OUT, exist_ok=True)

# ---------- 1. Network ----------
edges = pd.read_csv(os.path.join(DATA, "edge_list.csv"))
nodes = set(edges["from"]) | set(edges["to"])
parents, children = {}, {}
for _, r in edges.iterrows():
    children.setdefault(r["from"], []).append(r["to"])
    parents.setdefault(r["to"], []).append(r["from"])

sinks = [n for n in nodes if n not in children]

# Tier labels follow the inventory-parameter table of the paper
# (manuscript Table tab:inventory-params, App. C.4). A longest-path-from-sink
# BFS would put Memphis at depth 2 (same as Columbus/Richmond, i.e. Tier-4),
# but the paper labels Memphis as Tier-3 alongside Charlotte and Chicago.
TIER = {
    # Destination
    "NewYork":      0,
    # Last-mile DCs
    "Baltimore":    1,   # Tier-5 (LM)
    "Philadelphia": 1,   # Tier-5 (LM)
    # Tier-4
    "Columbus":     2,
    "Richmond":     2,
    # Tier-3 (Memphis included by paper convention; topological depth = 2)
    "Charlotte":    3,
    "Chicago":      3,
    "Memphis":      3,
    # Tier-2
    "Atlanta":      4,
    # Hub
    "Nashville":    5,
    # Sources -- no inflow on the released edges, dropped by compute_B below
    "SanFrancisco": 6,
    "StLouis":      6,
    "Orlando":      6,
}
missing = nodes - set(TIER)
assert not missing, f"TIER mapping missing nodes: {missing}"
tier = {n: TIER[n] for n in nodes}

print("Tiers:")
for t in sorted(set(tier.values())):
    print(f"  T{t}: {sorted(n for n in nodes if tier[n]==t)}")

# ---------- 2. Shipments → per-(node, item) daily inflow & outflow ----------
print("\nLoading shipments ...")
ship = pd.read_csv(
    os.path.join(DATA, "shipments.csv"),
    usecols=["day", "arrival_day", "from", "to", "item", "units"],
)
print(f"  shipments rows: {len(ship):,}")

# Outflow at node n on day t (item k): units shipped FROM n on day t
out = (
    ship.groupby(["from", "item", "day"])["units"].sum()
        .rename("units").reset_index()
        .rename(columns={"from": "node", "day": "t"})
)
# Inflow at node n on day t (item k): units arriving AT n on arrival_day t
inn = (
    ship.groupby(["to", "item", "arrival_day"])["units"].sum()
        .rename("units").reset_index()
        .rename(columns={"to": "node", "arrival_day": "t"})
)
del ship

# ---------- 3. Customer demand at retail sink (NewYork) ----------
dr = pd.read_csv(os.path.join(DATA, "daily_records.csv"),
                 usecols=["day", "item", "demand"])

# ---------- 4. Per-(node, item) variances ----------
T_MAX = int(dr["day"].max()) + 1
items = sorted(dr["item"].unique())
print(f"  horizon: {T_MAX} days, items: {len(items)}")

def to_dense(df, value_col="units"):
    """pivot a (node,item,t,value) frame to {node:{item: ndarray[T_MAX]}}."""
    out = {}
    for (node, item), g in df.groupby(["node", "item"]):
        v = np.zeros(T_MAX, dtype=np.float64)
        idx = g["t"].values.astype(int)
        # Some arrival days may exceed T_MAX (shipments still in transit at end).
        mask = idx < T_MAX
        v[idx[mask]] = g[value_col].values[mask]
        out.setdefault(node, {})[item] = v
    return out

print("Pivoting outflow ...")
outflow = to_dense(out)
print("Pivoting inflow ...")
inflow = to_dense(inn)

# Sink: outflow = customer demand
demand_NY = {}
for item, g in dr.groupby("item"):
    v = np.zeros(T_MAX, dtype=np.float64)
    v[g["day"].values.astype(int)] = g["demand"].values
    demand_NY[item] = v
sink = sinks[0]
outflow.setdefault(sink, {})
for item in items:
    outflow[sink][item] = demand_NY[item]   # override: customer-facing outflow

# ---------- 5. Compute B per (node, item) at a given aggregation window ----------
# Burn-in: drop first 365 days to avoid initialization transients
BURN = 365

def aggregate(series, window):
    """Sum a daily ndarray into non-overlapping `window`-day bins (drops trailing partial bin)."""
    if window <= 1:
        return series
    n = (len(series) // window) * window
    return series[:n].reshape(-1, window).sum(axis=1)

def compute_B(window, label, suffix):
    """Compute per-(node,item) bullwhip ratios at a given temporal aggregation."""
    rows = []
    for node in nodes:
        for item in items:
            d = outflow.get(node, {}).get(item)
            o = inflow.get(node, {}).get(item)
            if d is None or o is None:
                continue
            d, o = d[BURN:], o[BURN:]
            d, o = aggregate(d, window), aggregate(o, window)
            if len(d) < 2:
                continue
            vd, vo = d.var(ddof=1), o.var(ddof=1)
            if vd == 0 or vo == 0:
                continue
            rows.append(dict(
                node=node, tier=tier[node], item=item,
                var_demand=vd, var_inflow=vo, B=vo / vd,
                mean_demand=d.mean(), mean_inflow=o.mean(),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"bullwhip_per_node_item{suffix}.csv"), index=False)

    print(f"\n########## Aggregation: {label} (window={window}d, n_bins={(T_MAX-BURN)//window}) ##########")
    print(f"Per-(node,item) rows: {len(df)}")

    print(f"\n=== Per-node mean B ({label}) ===")
    node_summary = (
        df.groupby(["tier", "node"])
          .agg(B_mean=("B", "mean"),
               B_median=("B", "median"),
               B_p10=("B", lambda s: s.quantile(0.1)),
               B_p90=("B", lambda s: s.quantile(0.9)),
               n_items=("B", "count"))
          .reset_index()
          .sort_values(["tier", "node"])
    )
    print(node_summary.to_string(index=False))
    node_summary.to_csv(os.path.join(OUT, f"bullwhip_per_node{suffix}.csv"), index=False)

    print(f"\n=== Per-tier summary ({label}) ===")
    tier_summary = (
        df.groupby("tier")
          .agg(B_mean=("B", "mean"),
               B_median=("B", "median"),
               mean_var_demand=("var_demand", "mean"),
               mean_var_inflow=("var_inflow", "mean"),
               n_obs=("B", "count"))
          .reset_index()
          .sort_values("tier")
    )
    print(tier_summary.to_string(index=False))
    tier_summary.to_csv(os.path.join(OUT, f"bullwhip_per_tier{suffix}.csv"), index=False)

    print(f"\n=== Between-tier variance amplification ({label}) ===")
    btw = tier_summary[["tier", "mean_var_demand"]].copy()
    btw["amplification_to_next_upstream_tier"] = (
        btw["mean_var_demand"].shift(-1) / btw["mean_var_demand"]
    )
    print(btw.to_string(index=False))
    btw.to_csv(os.path.join(OUT, f"bullwhip_between_tier{suffix}.csv"), index=False)
    return df, node_summary, tier_summary

# Daily (original): suffix "" keeps backward-compatible filenames
compute_B(window=1,  label="daily",   suffix="")
# Monthly (Cachon-faithful, 30-day bins on simulation time)
compute_B(window=30, label="monthly", suffix="_monthly")

print("\nSaved:")
for f in ["bullwhip_per_node_item.csv","bullwhip_per_node.csv",
          "bullwhip_per_tier.csv","bullwhip_between_tier.csv",
          "bullwhip_per_node_item_monthly.csv","bullwhip_per_node_monthly.csv",
          "bullwhip_per_tier_monthly.csv","bullwhip_between_tier_monthly.csv"]:
    print(" ", os.path.join(OUT, f))

"""Generate baseline_overview.pdf for §3 from output_item50.

3 x 2 layout combining catalogue heterogeneity with mechanism:

  Left column (3 panels)  -- raw_item_series.png style: each row is
                             one catalogue item showing demand,
                             served, and unmet overlaid over the
                             full T = 52,560-day horizon. Items:
                             I01, I20, I40 sample the catalogue.

  Right column (3 panels) -- internal network state for the focal
                             item I01 in a 5-year zoom window
                             centred on the largest macro-shock:
                             (b) destination on-hand inventory
                             (d) destination backlog
                             (f) last-mile edge utilisation
                                 (PHL->NYC, BAL->NYC) with 0.95
                                 saturation threshold

The yellow band on each left panel marks the right column's window.
Together the two columns expose both the catalogue's per-item
heterogeneity and the mechanism that produces fill-rate drop events.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "output_item50"
OUT  = REPO / "results" / "figures"

# ---------- styling ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "lines.linewidth": 0.55,
})

C_DEMAND = "#3a7cb8"
C_SERVED = "#d97706"
C_UNMET = "#b91c1c"
C_ONHAND = "#0e7c66"
C_BACKLOG = "#7c3aed"
C_PHIL = "#c0392b"
C_BALT = "#e67e22"
C_THRESH = "#666666"
C_WIN = "#f1c40f"

ITEMS_LEFT = ["I01", "I02", "I03"]
FOCAL = "I01"

# ---------- load ----------
print("Loading daily records ...")
records = pd.read_csv(DATA / "daily_records.csv")
records["unmet"] = records["demand"] - records["served_from_stock"]

demand_pv = records.pivot(index="day", columns="item", values="demand")
served_pv = records.pivot(index="day", columns="item",
                          values="served_from_stock")
unmet_pv = records.pivot(index="day", columns="item", values="unmet")
onhand_pv = records.pivot(index="day", columns="item",
                           values="dest_on_hand_end_before_ship")
backlog_pv = records.pivot(index="day", columns="item",
                            values="dest_backlog_end_before_ship")
T = demand_pv.shape[0]
print(f"Horizon T = {T} days; left items: {ITEMS_LEFT}; "
      f"focal item: {FOCAL}")

# edge utilisation
edge_util = np.load(DATA / "edge_utilisation.npy")
edge_list = pd.read_csv(DATA / "edge_list.csv")
phil_idx = int(edge_list.index[(edge_list["from"] == "Philadelphia")
                               & (edge_list["to"] == "NewYork")][0])
balt_idx = int(edge_list.index[(edge_list["from"] == "Baltimore")
                               & (edge_list["to"] == "NewYork")][0])

# ---------- zoom window ----------
agg = demand_pv.sum(axis=1).values
WIN = 5 * 365
roll = pd.Series(agg).rolling(WIN, center=True).mean()
peak = int(roll.idxmax())
start = max(0, peak - WIN // 2)
end = min(T, start + WIN)
start = max(0, end - WIN)
print(f"Zoom window: days {start}..{end} (peak at day {peak})")

# focal-item slices
foc_onhand = onhand_pv[FOCAL].values
foc_backlog = backlog_pv[FOCAL].values
foc_onhand_z = foc_onhand[start:end]
foc_backlog_z = foc_backlog[start:end]
util_phil_z = edge_util[start:end, phil_idx]
util_balt_z = edge_util[start:end, balt_idx]

# ---------- figure ----------
fig, axes = plt.subplots(3, 2, figsize=(13, 7.4),
                          gridspec_kw={"hspace": 0.55, "wspace": 0.20,
                                       "width_ratios": [1.7, 1.0]})

panel_letters_left = ["a", "c", "e"]

# -------- left column: per-item demand/served/unmet --------
for r, item in enumerate(ITEMS_LEFT):
    d = demand_pv[item].values
    s = served_pv[item].values
    u = unmet_pv[item].values
    fr = s.sum() / max(d.sum(), 1e-9)
    ymax = max(d.max(), s.max()) * 1.10

    ax = axes[r, 0]
    ax.fill_between(np.arange(T), 0, u, color=C_UNMET, alpha=0.18,
                    lw=0)
    ax.plot(np.arange(T), d, color=C_DEMAND, lw=0.45, alpha=0.85,
            label="demand" if r == 0 else None)
    ax.plot(np.arange(T), s, color=C_SERVED, lw=0.45, alpha=0.85,
            label="served" if r == 0 else None)
    ax.plot(np.arange(T), u, color=C_UNMET, lw=0.45, alpha=0.95,
            label="unmet" if r == 0 else None)
    ax.axvspan(start, end, color=C_WIN, alpha=0.22, lw=0,
               label="zoom window" if r == 0 else None)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(f"{item} demand (units)")
    ax.set_title(f"({panel_letters_left[r]})  {item} full horizon "
                 f"($T={T}$, fill rate {fr:.3f})", loc="left")
    if r == 0:
        ax.legend(loc="upper right", ncol=4, frameon=False,
                  columnspacing=1.0, handlelength=1.4)
    if r == 2:
        ax.set_xlabel("time unit")

# -------- right column: focal-item mechanism in zoom --------
# y-axis scaled to the zoom-window range, not the full horizon, so
# the dynamics inside the window are readable
ymax_onh = max(1.0, foc_onhand_z.max() * 1.10)
ymax_bk = max(1.0, foc_backlog_z.max() * 1.10)
days_z = np.arange(start, end)

# (b) destination on-hand for focal item
ax = axes[0, 1]
ax.fill_between(days_z, 0, foc_onhand_z, color=C_ONHAND, alpha=0.18,
                lw=0)
ax.plot(days_z, foc_onhand_z, color=C_ONHAND, lw=0.85, alpha=0.95)
ax.set_xlim(start, end - 1)
ax.set_ylim(0, ymax_onh)
ax.set_ylabel(f"{FOCAL} on-hand")
ax.set_title(f"(b)  Zoom: {FOCAL} destination on-hand inventory",
             loc="left")

# (d) destination backlog for focal item
ax = axes[1, 1]
ax.fill_between(days_z, 0, foc_backlog_z, color=C_BACKLOG, alpha=0.18,
                lw=0)
ax.plot(days_z, foc_backlog_z, color=C_BACKLOG, lw=0.85, alpha=0.95)
ax.set_xlim(start, end - 1)
ax.set_ylim(0, ymax_bk)
ax.set_ylabel(f"{FOCAL} backlog")
ax.set_title(f"(d)  Zoom: {FOCAL} destination backlog", loc="left")

# (f) last-mile edge utilisation in zoom
ax = axes[2, 1]
ax.plot(days_z, util_phil_z, color=C_PHIL, lw=0.85, alpha=0.95,
        label=r"PHL$\to$NYC")
ax.plot(days_z, util_balt_z, color=C_BALT, lw=0.85, alpha=0.85,
        label=r"BAL$\to$NYC")
ax.axhline(0.95, color=C_THRESH, ls="--", lw=0.9, alpha=0.7,
           label=r"$U=0.95$")
ax.set_xlim(start, end - 1)
ax.set_ylim(0, 1.05)
ax.set_ylabel(r"$U_{e,t}$")
ax.set_xlabel("time unit")
ax.set_title("(f)  Zoom: last-mile edge utilisation", loc="left")
ax.legend(loc="upper left", ncol=3, frameon=False,
          columnspacing=1.0, handlelength=1.6)

plt.savefig(OUT / "baseline_overview.pdf", bbox_inches="tight")
plt.savefig(OUT / "baseline_overview.png", dpi=160,
            bbox_inches="tight")
print(f"Saved {OUT / 'baseline_overview.pdf'}")
print(f"Saved {OUT / 'baseline_overview.png'}")

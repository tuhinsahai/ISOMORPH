"""
Figure 4 (replacement): single-row small multiples, rendered as
two separate figures so each can be placed independently in the
paper.

For one item (I36, with high macro-shock sensitivity g_i and high
burst rate r_i), under four demand-side scenarios:

  scenario_family_demand.pdf
        realised demand y_{i,t}  +  deterministic intensity
        lambda_{i,t}  (Eq.\\ ref{eq:intensity})

  scenario_family_oh.pdf
        destination on-hand OH^{d*,i}_t at NewYork
        (end-of-step value; the network's response to demand)

Supply-side sweeps leave y_{i,t} unchanged by construction and
are omitted.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter
import matplotlib.patheffects as pe


def _short_count(x: float, _pos: int) -> str:
    """Compact integer formatter: 1.2M, 30K, 5K, 0."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e6:
        return f"{x/1e6:g}M"
    if ax >= 1e3:
        return f"{x/1e3:g}K"
    return f"{int(x)}"


SHORT = FuncFormatter(_short_count)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "output_mixture"
OUT  = REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ITEM = "I36"
T_FULL = 52560

# (dir name, panel title, accent colour for line, shadow tint)
SCENARIOS = [
    ("baseline",  r"baseline",
     "#3470a8", "#cddbeb"),  # cobalt  / pale blue
    ("drift_hi",  r"drift  $(\phi^{\mathrm{AR}}\!=\!0.99)$",
     "#7e5b9a", "#dcd2e5"),  # plum    / pale mauve
    ("shock_xhi", r"shock  $(N{\times}h^{G}\!=\!3{\times}4)$",
     "#c08438", "#f0d8b6"),  # caramel / pale cream
    ("burst_xhi", r"burst  $(r{\times}h^{P}\!=\!3{\times}4)$",
     "#a64141", "#e7c4c2"),  # brick   / pale rose
]

# ---------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "axes.titleweight": "normal",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.minor.size": 1.3,
    "ytick.minor.size": 1.3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})

# ---------------------------------------------------------------- data
def load_panel(scenario: str, item: str):
    df = pd.read_csv(
        DATA / scenario / "seed2025" / "daily_records.csv",
        usecols=("day", "item", "demand", "dest_on_hand_end_before_ship"),
    )
    s = df[df["item"] == item].sort_values("day")
    y = s["demand"].to_numpy()
    oh = s["dest_on_hand_end_before_ship"].to_numpy()
    assert y.size == T_FULL, f"{scenario}: got {y.size} rows for {item}"

    cols = (DATA / scenario / "seed2025" / "demand_signals_cols.txt").read_text().strip().split(",")
    j = cols.index(item)
    lam = np.load(DATA / scenario / "seed2025" / "demand_signals.npy")[:, j]
    return y, lam, oh


panels = [(scen, title, accent, shadow, *load_panel(scen, ITEM))
          for scen, title, accent, shadow in SCENARIOS]

# ---------------------------------------------------------------- figure
t_years = np.arange(T_FULL) / 1e4
decim = max(1, T_FULL // 6000)


# =================================================================
# Figure A: realised demand y_{i,t} + intensity lambda_{i,t}
# =================================================================
figA, axesA = plt.subplots(
    1, len(SCENARIOS), figsize=(12.0, 2.55),
    sharey=False,
    gridspec_kw={"wspace": 0.22},
)

for ax, (scen, title, accent, shadow, y, lam, _oh) in zip(axesA, panels):
    ax.plot(t_years[::decim], y[::decim],
            color=shadow, lw=0.35, alpha=0.85,
            zorder=1, rasterized=True)
    ax.plot(t_years, lam,
            color=accent, lw=0.95, alpha=1.0,
            solid_capstyle="round", solid_joinstyle="round",
            zorder=3,
            path_effects=[pe.Stroke(linewidth=1.9, foreground="white"),
                          pe.Normal()])
    ax.set_title(title, style="italic")
    ax.set_xlim(0, T_FULL / 1e4)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ymax_panel = max(y.max(), lam.max())
    ax.set_ylim(0, ymax_panel * 1.06)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=8, integer=True))

axesA[0].set_ylabel(rf"item {ITEM} demand")
figA.supxlabel(r"time $t$ ($\times 10^{4}$ time units)",
               y=0.04, fontsize=9.5)
figA.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.22)

outA = OUT / "scenario_family_demand.pdf"
figA.savefig(outA, bbox_inches="tight")
figA.savefig(outA.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(figA)


# =================================================================
# Figure B: destination (NewYork) on-hand OH^{d*, i}_t
# =================================================================
figB, axesB = plt.subplots(
    1, len(SCENARIOS), figsize=(12.0, 2.55),
    sharey=False,
    gridspec_kw={"wspace": 0.22},
)

for ax, (scen, title, accent, shadow, _y, _lam, oh) in zip(axesB, panels):
    ax.plot(t_years[::decim], oh[::decim],
            color=accent, lw=0.85, alpha=1.0,
            solid_capstyle="round", solid_joinstyle="round",
            zorder=3,
            path_effects=[pe.Stroke(linewidth=1.7, foreground="white"),
                          pe.Normal()])
    ax.set_title(title, style="italic")
    ax.set_xlim(0, T_FULL / 1e4)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ymax_panel = max(1.0, float(oh.max()))
    ax.set_ylim(0, ymax_panel * 1.08)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_formatter(SHORT)

axesB[0].set_ylabel(rf"destination on-hand, item {ITEM}")
figB.supxlabel(r"time $t$ ($\times 10^{4}$ time units)",
               y=0.04, fontsize=9.5)
figB.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.22)

outB = OUT / "scenario_family_oh.pdf"
figB.savefig(outB, bbox_inches="tight")
figB.savefig(outB.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(figB)


# =================================================================
# Figure C: combined 2x4 (demand row on top, on-hand row on bottom).
# Same data as Figures A and B, sharing the x-axis within each column.
# =================================================================
figC, axesC = plt.subplots(
    2, len(SCENARIOS), figsize=(12.0, 4.6),
    sharex="col", sharey=False,
    gridspec_kw={"wspace": 0.22, "hspace": 0.18},
)

for col, (scen, title, accent, shadow, y, lam, oh) in enumerate(panels):
    ax_top = axesC[0, col]
    ax_bot = axesC[1, col]

    ax_top.plot(t_years[::decim], y[::decim],
                color=shadow, lw=0.30, alpha=0.70,
                zorder=1, rasterized=True)
    ax_top.plot(t_years, lam,
                color=accent, lw=0.75, alpha=0.85,
                solid_capstyle="round", solid_joinstyle="round",
                zorder=3,
                path_effects=[pe.Stroke(linewidth=1.5, foreground="white"),
                              pe.Normal()])
    ax_top.set_title(title, style="italic")
    ax_top.set_xlim(0, T_FULL / 1e4)
    ax_top.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_top.xaxis.set_minor_locator(MultipleLocator(0.5))
    ymax_top = max(y.max(), lam.max())
    ax_top.set_ylim(0, ymax_top * 1.06)
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_top.yaxis.set_minor_locator(MaxNLocator(nbins=8, integer=True))

    ax_bot.plot(t_years[::decim], oh[::decim],
                color=accent, lw=0.75, alpha=0.85,
                solid_capstyle="round", solid_joinstyle="round",
                zorder=3,
                path_effects=[pe.Stroke(linewidth=1.5, foreground="white"),
                              pe.Normal()])
    ax_bot.set_xlim(0, T_FULL / 1e4)
    ax_bot.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_bot.xaxis.set_minor_locator(MultipleLocator(0.5))
    ymax_bot = max(1.0, float(oh.max()))
    ax_bot.set_ylim(0, ymax_bot * 1.08)
    ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_bot.yaxis.set_minor_locator(MaxNLocator(nbins=8, integer=True))
    ax_bot.yaxis.set_major_formatter(SHORT)

axesC[0, 0].set_ylabel(rf"item {ITEM} demand")
axesC[1, 0].set_ylabel(rf"destination on-hand, item {ITEM}")
figC.supxlabel(r"time $t$ ($\times 10^{4}$ time units)",
               y=0.02, fontsize=9.5)
figC.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.11)

outC = OUT / "scenario_family.pdf"
figC.savefig(outC, bbox_inches="tight")
figC.savefig(outC.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(figC)

print(f"wrote {outA}, {outB}, and {outC}  item={ITEM}  "
      f"scenarios={[s for s, *_ in SCENARIOS]}")

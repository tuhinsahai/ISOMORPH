"""§4 UQ forecast-envelope figure: parameter UQ propagated to forecaster output.

For each model with K=20 LHS perturbation tensors, plots median + 10/90
band of y_true (input) and y_pred (output) across K, at one or three
forecast windows for one item. The grey band is the band of physical
realisations the network produces under demand-side parameter perturbation;
the coloured band is the band of zero-shot forecasts of those realisations.

  python plot_uq_envelope.py                          # 2x2, deterministic mid window
  python plot_uq_envelope.py --multi                  # 3x4 multi-window grid
  python plot_uq_envelope.py --window 25              # explicit single window
  python plot_uq_envelope.py --item I05 chronos       # subset of models
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO / "results" / "eval" / "uq"
FIG_DIR = REPO / "results" / "uq" / "figures"

MODELS = {
    "chronos":  {"prefix": "chronos_t5_base",
                 "fill": "#9BB0CC", "line": "#2F4A75",
                 "display": "Chronos"},
    "moirai":   {"prefix": "moirai_1_1_R_base",
                 "fill": "#A8BFA0", "line": "#345531",
                 "display": "Moirai"},
    "timesfm":  {"prefix": "timesfm_2_0_500m_pytorch",
                 "fill": "#D7A992", "line": "#693220",
                 "display": "TimesFM"},
    "lagllama": {"prefix": "lag_llama",
                 "fill": "#C2A8CC", "line": "#4D2752",
                 "display": "Lag-Llama"},
}
TRUTH_BAND_COLOR = "#9CA3AF"
TRUTH_LINE_COLOR = "#374151"
AXES_FACE = "#FFFFFF"
SPINE_COLOR = "#374151"


def load_K_tensors(prefix: str, K: int = 20):
    yp_list, yt_list = [], []
    item_ids = window_starts = None
    for k in range(1, K + 1):
        p = RESULT_DIR / f"{prefix}_perturb_k{k:02d}_tensors.npz"
        if not p.exists():
            return None
        d = np.load(p)
        yp_list.append(d["y_pred"])
        yt_list.append(d["y_true"])
        if item_ids is None:
            item_ids, window_starts = d["item_ids"], d["window_starts"]
    y_pred = np.stack(yp_list, axis=0)   # (K, W, H, C)
    y_true = np.stack(yt_list, axis=0)
    return y_pred, y_true, item_ids, window_starts


def deterministic_windows(W: int, n: int) -> list[int]:
    """Evenly-spaced windows inside the test split, avoiding the edges."""
    if n == 1:
        return [W // 2]
    return [int(round(W * (i + 1) / (n + 1))) for i in range(n)]


def draw_band(ax, arr_kh: np.ndarray, fill_color: str, line_color: str,
              label: str, fill_alpha: float, lw: float, z_base: int):
    h = np.arange(1, arr_kh.shape[1] + 1)
    med = np.median(arr_kh, axis=0)
    q10 = np.percentile(arr_kh, 10, axis=0)
    q90 = np.percentile(arr_kh, 90, axis=0)
    ax.fill_between(h, q10, q90, color=fill_color, alpha=fill_alpha,
                    linewidth=0, zorder=z_base)
    ax.plot(h, med, color=line_color, lw=lw, label=label, zorder=z_base + 2)


def style_axes(ax):
    ax.set_facecolor(AXES_FACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=SPINE_COLOR, length=3, width=0.7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(SPINE_COLOR)
    ax.grid(True, linestyle=':', alpha=0.35, linewidth=0.5,
            color=SPINE_COLOR)


def _draw_main(ax, yp_kh, yt_kh, model_key, draw_legend):
    draw_band(ax, yt_kh,
              fill_color=TRUTH_BAND_COLOR, line_color=TRUTH_LINE_COLOR,
              label=r"truth $y_{i,t}$",
              fill_alpha=0.30, lw=1.0, z_base=1)
    draw_band(ax, yp_kh,
              fill_color=MODELS[model_key]["fill"],
              line_color=MODELS[model_key]["line"],
              label=r"forecast $\hat y_{i,t}$",
              fill_alpha=0.30, lw=1.8, z_base=3)
    style_axes(ax)
    if draw_legend:
        leg = ax.legend(loc="upper left", fontsize=8.5, frameon=True,
                        facecolor=AXES_FACE, edgecolor=SPINE_COLOR)
        leg.get_frame().set_linewidth(0.6)
        for txt in leg.get_texts():
            txt.set_color(SPINE_COLOR)


def _zoom_ylim(yp_kh, yt_kh, pad_frac: float = 0.08):
    yt_med = np.median(yt_kh, axis=0)
    yp_med = np.median(yp_kh, axis=0)
    y_lo = float(min(yt_med.min(), yp_med.min()))
    y_hi = float(max(yt_med.max(), yp_med.max()))
    span = max(y_hi - y_lo, 1e-6)
    pad = pad_frac * span
    return y_lo - pad, y_hi + pad


def _draw_zoom(zoom_ax, main_ax, yp_kh, yt_kh, model_key):
    """Sibling axes below `main_ax` with the same bands and medians, but
    y-axis tightened to the median range. Also shades the corresponding
    horizontal slice on `main_ax` so the link is explicit.
    """
    h = np.arange(1, yp_kh.shape[1] + 1)
    yt_med = np.median(yt_kh, axis=0)
    yp_med = np.median(yp_kh, axis=0)
    yt_q10 = np.percentile(yt_kh, 10, axis=0)
    yt_q90 = np.percentile(yt_kh, 90, axis=0)
    yp_q10 = np.percentile(yp_kh, 10, axis=0)
    yp_q90 = np.percentile(yp_kh, 90, axis=0)
    y_lo, y_hi = _zoom_ylim(yp_kh, yt_kh)

    zoom_ax.fill_between(h, yt_q10, yt_q90, color=TRUTH_BAND_COLOR,
                         alpha=0.30, linewidth=0, zorder=1)
    zoom_ax.plot(h, yt_med, color=TRUTH_LINE_COLOR, lw=1.0, zorder=3)
    zoom_ax.fill_between(h, yp_q10, yp_q90,
                         color=MODELS[model_key]["fill"],
                         alpha=0.30, linewidth=0, zorder=2)
    zoom_ax.plot(h, yp_med, color=MODELS[model_key]["line"], lw=1.6, zorder=4)
    zoom_ax.set_xlim(int(h[0]), int(h[-1]))
    zoom_ax.set_ylim(y_lo, y_hi)
    style_axes(zoom_ax)
    zoom_ax.tick_params(axis="y", labelsize=8)

    # Mark the zoom y-slice on the parent so the reader sees exactly which
    # part of the main panel is being zoomed.
    main_ax.axhspan(y_lo, y_hi, color=SPINE_COLOR, alpha=0.10,
                    linewidth=0, zorder=0.5)
    main_ax.axhline(y_lo, color=SPINE_COLOR, lw=0.5, ls=":",
                    alpha=0.7, zorder=0.6)
    main_ax.axhline(y_hi, color=SPINE_COLOR, lw=0.5, ls=":",
                    alpha=0.7, zorder=0.6)


def plot_2x2(data: dict, item_id: str, item_idx: int, w: int,
             window_start: int, fig_path: Path, with_zoom: bool = True):
    fig_h = 8.4 if with_zoom else 5.6
    fig = plt.figure(figsize=(9.6, fig_h))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.18,
                             left=0.07, right=0.99, top=0.94, bottom=0.07)
    items = list(data.items())
    placements = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (ri, ci), (model_key, (yp, yt)) in zip(placements, items):
        yp_kh = yp[:, w, :, item_idx]
        yt_kh = yt[:, w, :, item_idx]
        if with_zoom:
            inner = outer[ri, ci].subgridspec(
                2, 1, height_ratios=[2.6, 1.9], hspace=0.06)
            main_ax = fig.add_subplot(inner[0])
            zoom_ax = fig.add_subplot(inner[1], sharex=main_ax)
        else:
            main_ax = fig.add_subplot(outer[ri, ci])
            zoom_ax = None
        _draw_main(main_ax, yp_kh, yt_kh, model_key,
                   draw_legend=(ri == 0 and ci == 0))
        main_ax.set_title(MODELS[model_key]["display"], fontsize=11,
                          color=SPINE_COLOR)
        if zoom_ax is not None:
            _draw_zoom(zoom_ax, main_ax, yp_kh, yt_kh, model_key)
            plt.setp(main_ax.get_xticklabels(), visible=False)
        if ri == 1:
            (zoom_ax if zoom_ax is not None else main_ax).set_xlabel(
                r"forecast horizon $h$ (time units)", color=SPINE_COLOR)
        if ci == 0:
            main_ax.set_ylabel(f"item {item_id} demand",
                               color=SPINE_COLOR)
            if zoom_ax is not None:
                zoom_ax.set_ylabel("zoom (medians)", fontsize=8.5,
                                   color=SPINE_COLOR)
    fig.savefig(fig_path, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight",
                dpi=160, facecolor="white")
    plt.close(fig)


def plot_multi(data: dict, item_id: str, item_idx: int,
               windows: list[int], window_starts_arr: np.ndarray,
               fig_path: Path):
    """Grid: rows = windows, cols = models. Each cell is a (main, zoom)
    vertical pair sharing x; the zoom row uses tightened y-limits."""
    n_rows = len(windows)
    n_cols = len(data)
    fig = plt.figure(figsize=(3.2 * n_cols, 4.0 * n_rows))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.32, wspace=0.20,
                             left=0.06, right=0.99, top=0.95, bottom=0.06)
    model_items = list(data.items())
    for r, w in enumerate(windows):
        t0 = int(window_starts_arr[w])
        for c, (model_key, (yp, yt)) in enumerate(model_items):
            inner = outer[r, c].subgridspec(
                2, 1, height_ratios=[2.6, 1.9], hspace=0.06)
            main_ax = fig.add_subplot(inner[0])
            zoom_ax = fig.add_subplot(inner[1], sharex=main_ax)
            yp_kh = yp[:, w, :, item_idx]
            yt_kh = yt[:, w, :, item_idx]
            _draw_main(main_ax, yp_kh, yt_kh, model_key,
                       draw_legend=(r == 0 and c == 0))
            if r == 0:
                main_ax.set_title(MODELS[model_key]["display"], fontsize=11,
                                  color=SPINE_COLOR)
            _draw_zoom(zoom_ax, main_ax, yp_kh, yt_kh, model_key)
            plt.setp(main_ax.get_xticklabels(), visible=False)
            if c == 0:
                main_ax.set_ylabel(f"$t_0{{=}}{t0}$", fontsize=10,
                                   color=SPINE_COLOR)
                zoom_ax.set_ylabel("zoom", fontsize=8.5, color=SPINE_COLOR)
            if r == n_rows - 1:
                zoom_ax.set_xlabel(r"forecast horizon $h$ (time units)",
                                   color=SPINE_COLOR)
    fig.text(0.005, 0.5, f"item {item_id} demand",
             rotation="vertical", va="center", ha="left",
             fontsize=10.5, color=SPINE_COLOR)
    fig.savefig(fig_path, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight",
                dpi=160, facecolor="white")
    plt.close(fig)


def main():
    args = sys.argv[1:]
    item = "I01"
    window_arg: int | None = None
    multi = False
    narrowest = False
    if "--item" in args:
        i = args.index("--item"); item = args[i + 1]; args = args[:i] + args[i + 2:]
    if "--window" in args:
        i = args.index("--window"); window_arg = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if "--multi" in args:
        multi = True; args.remove("--multi")
    if "--narrowest" in args:
        narrowest = True; args.remove("--narrowest")
    no_zoom = False
    if "--no-zoom" in args:
        no_zoom = True; args.remove("--no-zoom")
    requested = args if args else list(MODELS.keys())
    bad = [m for m in requested if m not in MODELS]
    if bad:
        sys.exit(f"unknown model(s): {bad}; choose from {list(MODELS.keys())}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    item_ids = window_starts = None
    for m in requested:
        out = load_K_tensors(MODELS[m]["prefix"])
        if out is None:
            print(f"[{m}] missing tensors, skipping")
            continue
        yp, yt, ids, ws = out
        if item_ids is None:
            item_ids, window_starts = ids, ws
        data[m] = (yp, yt)
        print(f"[{m}] y_pred={yp.shape}, y_true={yt.shape}")
    if not data:
        sys.exit("no models loaded")

    item_idx = list(item_ids).index(item)
    W_total = next(iter(data.values()))[0].shape[1]

    if multi:
        windows = deterministic_windows(W_total, 3)
        print(f"item {item} (idx={item_idx}); multi-window grid w={windows} "
              f"(t0={[int(window_starts[w]) for w in windows]})")
        out = FIG_DIR / f"uq_envelope_{item}_multi.pdf"
        plot_multi(data, item, item_idx, windows,
                   np.asarray(window_starts), out)
        print(f"wrote {out}")
        return

    if window_arg is None:
        if narrowest:
            yt_any = next(iter(data.values()))[1]
            win_mean = yt_any[:, :, :, item_idx].mean(axis=-1)  # (K, W)
            spread = win_mean.std(axis=0)                       # (W,)
            w = int(spread.argmin())
            print(f"min cross-K spread window: w={w}, "
                  f"spread={spread[w]:.2f} "
                  f"(min={spread.min():.2f}, max={spread.max():.2f})")
        else:
            w = 12
            print(f"default window: w={w}")
    else:
        w = window_arg
    t0 = int(window_starts[w])
    print(f"item {item} (idx={item_idx}); window w={w}, t_start={t0}")

    if window_arg is not None:
        suffix = f"{item}_w{w:02d}"
    elif narrowest:
        suffix = f"{item}_narrowest"
    else:
        suffix = item
    if no_zoom:
        suffix = f"{suffix}_nozoom"
    out = FIG_DIR / f"uq_envelope_{suffix}.pdf"
    plot_2x2(data, item, item_idx, w, t0, out, with_zoom=not no_zoom)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

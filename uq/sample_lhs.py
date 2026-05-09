"""Latin hypercube sample of K=20 demand-side perturbation configurations
for the §4.4 UQ experiment.

Three demand-side knobs, perturbed jointly around the §4 baseline:
  phi_AR  in [0.95, 0.999]   AR(1) drift coefficient (used directly)
  rho_G   in [0.5,  2.0  ]   macro-shock multiplier (applied jointly to
                             shock_count_scale and shock_height_scale)
  rho_B   in [0.5,  2.0  ]   burst multiplier (applied jointly to
                             burst_rate_scale and burst_height_scale)

Writes output_uq/manifest.csv with columns (k, phi_AR, rho_G, rho_B).
The submit script reads this and launches one simulator job per row.

Synchronous scaling for shock and burst matches the §3.3 axis convention
(shock axis = N x h^G; burst axis = r x h^P). Drift is a single scalar
per run, replacing the per-item U[phi_lo, phi_hi] draw with phi_lo =
phi_hi = phi_AR_k (matching the §3.3 drift sweep convention).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


K = 20
SEED = 2025

# (lo, hi) for each of the three demand-side knobs
RANGES = {
    "phi_AR": (0.95, 0.999),
    "rho_G":  (0.5,  2.0),
    "rho_B":  (0.5,  2.0),
}

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "output_uq"


def main() -> None:
    sampler = qmc.LatinHypercube(d=len(RANGES), seed=SEED)
    unit = sampler.random(n=K)                         # (K, 3) in [0, 1)

    los = np.array([RANGES[k][0] for k in RANGES])
    his = np.array([RANGES[k][1] for k in RANGES])
    scaled = los + unit * (his - los)                  # (K, 3) in ranges

    df = pd.DataFrame({
        "k":      np.arange(1, K + 1),
        "phi_AR": np.round(scaled[:, 0], 6),
        "rho_G":  np.round(scaled[:, 1], 6),
        "rho_B":  np.round(scaled[:, 2], 6),
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(df.to_string(index=False))

    print("\nRange checks:")
    for col in ["phi_AR", "rho_G", "rho_B"]:
        lo, hi = RANGES[col]
        print(f"  {col:7s}: [{df[col].min():.4f}, {df[col].max():.4f}]  "
              f"(target [{lo}, {hi}])")


if __name__ == "__main__":
    main()

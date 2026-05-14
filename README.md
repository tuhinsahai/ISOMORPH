# ISOMORPH

A digital twin of a multi-echelon logistics network, plus a zero-shot
foundation-model evaluation harness. This repository accompanies the
paper *ISOMORPH: A Supply Chain Digital Twin for Simulation, Dataset
Generation, and Forecasting Benchmarks*:
https://arxiv.org/pdf/2605.12768

The release contains four pieces:

1. The simulator that produces every released dataset
   (`simulator/`).
2. Zero-shot rolling-origin inference and metric scripts for four TSF
   foundation models: Chronos, Moirai, TimesFM, Lag-Llama (`eval/`).
3. The Latin-hypercube parameter-uncertainty pipeline used for the
   forward-UQ figures (`uq/`).
4. Validation and figure scripts (bullwhip, baseline overview, scenario
   family) (`analysis/`).

The released datasets are distributed alongside this code in a sibling
`data/` directory; see *Data layout* below.

## Layout

```
Isomorph_release/
├── README.md
├── requirements.txt
├── simulator/
│   ├── Supplychaingeo_item50.py     # canonical simulator, C=50 catalogue
│   ├── Supplychaingeo_item200.py    # same logic, C=200 catalogue
│   └── derive_edge_files.py         # post-process: edge_list.csv + utilisation
├── eval/
│   ├── {chronos,moirai,timesfm,lagllama}_runner.py   # model wrappers
│   ├── {chronos,moirai,timesfm,lagllama}_run.py      # CLI driver per model
│   ├── data_utils.py                # load_dataset, iter_test_windows
│   ├── metrics.py                   # per-channel MAE/RMSE at horizons
│   └── gift_style_mase.py           # paper Tables 3 / 13 aggregator
├── uq/
│   ├── sample_lhs.py                # K=20 LHS over (phi_AR, rho_G, rho_B)
│   └── plot_uq_envelope.py          # forecast-envelope figure
└── analysis/
    ├── bullwhip_analysis.py         # paper Tables 2 and 18
    ├── make_baseline_overview.py    # paper Figure 6
    └── make_scenario_family.py      # paper Figure 3
```

## Data layout

The released datasets live in a sibling `data/` directory and are
referenced by every CLI driver via `--root <data dir>`:

```
data/
├── output_item50/                # baseline C=50
├── output_item200/               # baseline C=200
├── output_mixture/<scenario>/    # 27 scenario rollouts at C=50
└── output_uq/
    ├── manifest.csv              # K=20 LHS configurations
    └── perturb_k01 ... k20/      # one rollout per LHS sample
```

Each rollout writes:
- `daily_records.csv`, `shipments.csv`, `service_summary.csv`
- `inventory_history.csv`, `backlog_history.csv`, `intransit_history.csv`
- `demand_signals.npy`, `demand_signals_cols.txt`
- `scenario.json` (the exact CLI knobs used)
- optional: `edge_list.csv`, `edge_utilisation.npy`, `edge_saturation.npy`
  (produced by `simulator/derive_edge_files.py`)

## Conventions

- One step is one day. The released horizon is `T = 52,560`.
- All released runs use seed `2025`.

---

## 1. Generate the baseline datasets

C = 50 items:
```bash
python simulator/Supplychaingeo_item50.py \
    --days 52560 --seed 2025 \
    --pipeline_mult 7 \
    --out_dir data/output_item50 \
    --scenario_name baseline
```

C = 200 items:
```bash
python simulator/Supplychaingeo_item200.py \
    --days 52560 --seed 2025 \
    --pipeline_mult 7 \
    --out_dir data/output_item200
```

Both use horizon `T = 52,560` and pipeline multiplier `m = 7`. The
released datasets ship with `m = 7`; the simulator's built-in default
is `m = 0`.

## 2. Generate the scenario sweeps

The mixture set covers the six one-at-a-time sweeps of the paper plus
two compound scenarios. All sweeps run on the C=50 simulator. Each
named scenario is reproduced by overriding the corresponding knobs
below; the remaining knobs stay at their baseline.

| Sweep      | Knobs perturbed                              | Settings                                              |
|------------|----------------------------------------------|-------------------------------------------------------|
| Drift      | `--phi_lo`, `--phi_hi`                       | `0.71, 0.86, 0.96, 0.99, 0.9993`                      |
| Shock      | `--shock_count_scale`, `--shock_height_scale`| `(0,1), (0.5,0.7), (1,1), (2,2), (3,4)`               |
| Burst      | `--burst_rate_scale`, `--burst_height_scale` | `(1,1), (1.5,2), (2,3), (3,4), (5,8)`                 |
| Edge cap   | `--containers_scale`                         | `0.3, 0.6, 1.0, 1.5, 2.5`                             |
| Buffer     | `--ss_scale`                                 | `0.1, 0.2, 0.5, 0.75, 1.0`                            |
| Lead time  | `--leadtime_scale`                           | `1.0, 2.0, 5.0, 10.0, 20.0`                           |

Two compound scenarios used in the foundation-model evaluation:

| Scenario        | Overrides                                                              |
|-----------------|------------------------------------------------------------------------|
| `chaos_compound`| `phi_lo=0.96, phi_hi=0.98, shock_count_scale=3, shock_height_scale=4`  |
| `chaos_burst`   | `phi_lo=0.96, phi_hi=0.98, burst_rate_scale=3, burst_height_scale=4`   |

Example (drift_mid):
```bash
python simulator/Supplychaingeo_item50.py \
    --days 52560 --seed 2025 --pipeline_mult 7 \
    --phi_lo 0.95 --phi_hi 0.97 \
    --out_dir data/output_mixture/drift_mid \
    --scenario_name drift_mid
```

## 3. Foundation-model zero-shot evaluation

Each model has a thin wrapper (`*_runner.py`) and a CLI driver
(`*_run.py`) that performs rolling-origin inference and writes
per-channel metrics. All four models share `L=512`, `H=30`,
`stride=30`, `num_samples=20` (TimesFM uses its deterministic quantile
head). The paper's MASE is the GIFT-Eval-style aggregate computed by
`eval/gift_style_mase.py`.

Run one model on one dataset:
```bash
python eval/chronos_run.py \
    --root data \
    --dataset output_item50 \
    --model_id amazon/chronos-t5-base \
    --L 512 --H 30 --stride 30 \
    --num_samples 20 --channel_batch 16 \
    --out results/eval/baseline_and_scenarios
```

Substitute `chronos_run.py` with `moirai_run.py`, `timesfm_run.py`, or
`lagllama_run.py`, and the corresponding `--model_id`:

| Driver           | `--model_id`                                       |
|------------------|----------------------------------------------------|
| `chronos_run.py` | `amazon/chronos-t5-base`                           |
| `moirai_run.py`  | `Salesforce/moirai-1.1-R-base`                     |
| `timesfm_run.py` | `google/timesfm-2.0-500m-pytorch`                  |
| `lagllama_run.py`| `time-series-foundation-models/Lag-Llama`          |

To run on a scenario rollout, point
`--dataset output_mixture/<scenario>` and pass
`--label <scenario>`.

Aggregate to GIFT-Eval-style MASE (paper Tables 3 and 13):
```bash
python eval/gift_style_mase.py
```

## 4. Forward UQ (forecast envelopes)

Sample K=20 demand-side LHS configurations:
```bash
python uq/sample_lhs.py
```

This writes `data/output_uq/manifest.csv` with three knobs per row
(`phi_AR`, `rho_G`, `rho_B`). Run the simulator once per row:
```bash
while IFS=, read -r k phi rho_G rho_B; do
    [ "$k" = "k" ] && continue
    python simulator/Supplychaingeo_item50.py \
        --days 52560 --seed 2025 --pipeline_mult 7 \
        --phi_lo "$phi" --phi_hi "$phi" \
        --shock_count_scale "$rho_G" --shock_height_scale "$rho_G" \
        --burst_rate_scale  "$rho_B" --burst_height_scale  "$rho_B" \
        --out_dir "data/output_uq/perturb_k$(printf %02d $k)" \
        --scenario_name "perturb_k$(printf %02d $k)"
done < data/output_uq/manifest.csv
```

Run zero-shot inference on each rollout. Use the same `*_run.py`
drivers as in §3, but point `--dataset` at the per-perturbation
directory and override `--out` to a UQ-specific path:
```bash
for k in $(seq -f %02g 1 20); do
    python eval/chronos_run.py \
        --root data \
        --dataset "output_uq/perturb_k${k}" \
        --label "perturb_k${k}" \
        --out results/eval/uq
done
```
Repeat with `moirai_run.py` / `timesfm_run.py` / `lagllama_run.py` and
their `--model_id` (see §3). Then plot the forecast envelopes (paper
Figures 4 and 7):
```bash
python uq/plot_uq_envelope.py             # single window (Figure 4)
python uq/plot_uq_envelope.py --multi     # 3x4 multi-window (Figure 7)
```

## 5. Validation and figures

Bullwhip ratios per node and per tier:
```bash
python analysis/bullwhip_analysis.py
```

Baseline overview and scenario family figures:
```bash
python analysis/make_baseline_overview.py
python analysis/make_scenario_family.py
```

## Environment

The runs in the paper used Python 3.12, PyTorch with CUDA, and a single
NVIDIA RTX 2080 Ti. Install dependencies with:

```bash
pip install -r requirements.txt
```

A single 2080 Ti is sufficient for the longest run (Lag-Llama at
`L=512` finishes in under 5 hours per dataset).


## Acknowledgements

This material is based upon work of the authors supported by the Defense
Advanced Research Projects Agency (DARPA) under Agreement No.
HR00112590112. Approved for public release; distribution is unlimited.


## Citation

If you use this repository or dataset, please cite:

```bibtex
@misc{zhang2026isomorphsupplychaindigital,
      title={ISOMORPH: A Supply Chain Digital Twin for Simulation, Dataset Generation, and Forecasting Benchmarks}, 
      author={Zhizhen Zhang and Hyemin Gu and Benjamin J. Zhang and Daniel Elenius and Michael Tyrrell and Theo J. Bourdais and Houman Owhadi and Markos A. Katsoulakis and Tuhin Sahai},
      year={2026},
      eprint={2605.12768},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.12768}, 
}
```

## Licence

- Code: MIT.
- Outputs generated by running the scripts (datasets, figures): CC-BY-4.0.

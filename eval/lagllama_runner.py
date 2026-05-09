"""
Lag-Llama zero-shot rolling-origin inference.

Wraps time-series-foundation-models/Lag-Llama via the official GluonTS-style
LagLlamaEstimator. Univariate (like Chronos / TimesFM): per window we hand
the model C 1-D context arrays and reduce 100 sample paths to the per-day
median for the point forecast.

CRITICAL: Lag-Llama was trained on context length 32. Any L > 32 requires
RoPE scaling (linear factor = (L + H) / 32). Without this the position
embeddings extrapolate and the forecast degrades sharply.

Designed to mirror chronos_runner.py's I/O contract so metrics.py stays
unchanged.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset

from huggingface_hub import hf_hub_download
from lag_llama.gluon.estimator import LagLlamaEstimator


_LAGLLAMA_TRAINING_CTX = 32   # what the public checkpoint was trained on


def load_lagllama(prediction_length: int, context_length: int,
                  num_samples: int = 100, batch_size: int = 32,
                  ckpt_path: str | None = None,
                  device: str | None = None):
    """Load Lag-Llama as a GluonTS PyTorchPredictor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt_path is None:
        ckpt_path = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt",
        )
    print(f"  loading Lag-Llama from {ckpt_path}  on {device}  "
          f"(L={context_length}, H={prediction_length}, "
          f"num_samples={num_samples}, batch_size={batch_size})",
          file=sys.stderr)

    # Pull the model architecture knobs out of the checkpoint so we
    # construct a matching estimator regardless of release.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["hyper_parameters"]["model_kwargs"]

    # RoPE scaling so positional encodings extrapolate cleanly to L > 32.
    rope_scaling = None
    needed_extent = context_length + prediction_length
    if needed_extent > _LAGLLAMA_TRAINING_CTX:
        rope_scaling = {
            "type": "linear",
            "factor": float(needed_extent) / float(_LAGLLAMA_TRAINING_CTX),
        }
        print(f"  enabling RoPE scaling: factor="
              f"{rope_scaling['factor']:.3f}  "
              f"(L+H={needed_extent} > training ctx {_LAGLLAMA_TRAINING_CTX})",
              file=sys.stderr)

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=args["input_size"],
        n_layer=args["n_layer"],
        n_embd_per_head=args["n_embd_per_head"],
        n_head=args["n_head"],
        scaling=args["scaling"],
        time_feat=args["time_feat"],
        rope_scaling=rope_scaling,
        batch_size=batch_size,
        num_parallel_samples=num_samples,
    )
    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor


def predict_rolling_origin(predictor, D: np.ndarray,
                           window_starts: list[int],
                           L: int, H: int) -> np.ndarray:
    """Returns y_pred of shape (n_windows, H, n_items); point=median."""
    n_windows = len(window_starts)
    n_items = D.shape[1]
    y_pred = np.zeros((n_windows, H, n_items), dtype=np.float32)
    anchor = pd.Period("2000-01-01", freq="D")

    t0 = time.time()
    for wi, t in enumerate(window_starts):
        ctx = D[t - L:t, :]                                 # (L, n_items)
        items = [
            {"target": ctx[:, j].astype(np.float32), "start": anchor}
            for j in range(n_items)
        ]
        ds = ListDataset(items, freq="D")
        forecasts = list(predictor.predict(ds))
        for j, fc in enumerate(forecasts):
            # fc.samples: (num_parallel_samples, H)
            y_pred[wi, :, j] = np.median(
                np.asarray(fc.samples, dtype=np.float32), axis=0
            )
        if wi == 0 or (wi + 1) % 10 == 0 or wi + 1 == n_windows:
            elapsed = time.time() - t0
            rate = (wi + 1) / max(elapsed, 1e-9)
            eta = (n_windows - wi - 1) / max(rate, 1e-9)
            print(f"  window {wi+1:4d}/{n_windows}  "
                  f"elapsed={elapsed:6.1f}s  "
                  f"rate={rate:5.2f} win/s  "
                  f"eta={eta/60:5.1f}min", file=sys.stderr)
    return y_pred


def collect_y_true(D: np.ndarray, window_starts: list[int],
                   H: int) -> np.ndarray:
    n_windows = len(window_starts)
    n_items = D.shape[1]
    y_true = np.zeros((n_windows, H, n_items), dtype=np.float32)
    for wi, t in enumerate(window_starts):
        y_true[wi] = D[t:t + H]
    return y_true

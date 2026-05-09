"""
TimesFM zero-shot rolling-origin inference.

Wraps google/timesfm-2.0-500m-pytorch (univariate, like Chronos). Per
window we hand the model a list of C univariate context arrays and it
returns a (C, horizon_len) median-point forecast that we slice to H=30.

Designed to mirror chronos_runner.py's I/O contract so metrics.py can
stay unchanged.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

import timesfm


def load_timesfm(model_id: str = "google/timesfm-2.0-500m-pytorch",
                 context_len: int = 2048, horizon_len: int = 128,
                 per_core_batch_size: int = 32):
    """Build a TimesFM v2 predictor (PyTorch backend, GPU-aware)."""
    backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"  loading {model_id}  on {backend}  "
          f"(L={context_len}, horizon_len={horizon_len}, "
          f"per_core_batch_size={per_core_batch_size})", file=sys.stderr)

    # v2-500m-pytorch hparams (per timesfm finetuning_example).
    hparams = timesfm.TimesFmHparams(
        backend=backend,
        per_core_batch_size=per_core_batch_size,
        horizon_len=horizon_len,
        num_layers=50,
        use_positional_embedding=False,
        context_len=context_len,
        # defaults: model_dims=1280, num_heads=16,
        # input_patch_len=32, output_patch_len=128,
        # quantiles=(.1,.2,...,.9), point_forecast_mode='median'
    )
    tfm = timesfm.TimesFm(
        hparams=hparams,
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id),
    )
    return tfm


def predict_rolling_origin(tfm, D: np.ndarray, window_starts: list[int],
                           L: int, H: int) -> np.ndarray:
    """Returns y_pred of shape (n_windows, H, n_items); point=median."""
    n_windows = len(window_starts)
    n_items = D.shape[1]
    y_pred = np.zeros((n_windows, H, n_items), dtype=np.float32)

    t0 = time.time()
    for wi, t in enumerate(window_starts):
        ctx = D[t - L:t, :]                    # (L, n_items)
        # forecast() takes a list of 1-D arrays, one per channel.
        inputs = [ctx[:, j].astype(np.float32) for j in range(n_items)]
        # freq=0 ("high frequency") matches daily; harmless either way for
        # zero-shot eval since TimesFM uses freq only as a coarse covariate.
        freqs = [0] * n_items
        point_fc, _ = tfm.forecast(inputs=inputs, freq=freqs)
        # point_fc shape: (n_items, horizon_len). Slice to first H steps.
        y_pred[wi] = np.asarray(point_fc[:, :H], dtype=np.float32).T
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
    """Mirror chronos_runner.collect_y_true."""
    n_windows = len(window_starts)
    n_items = D.shape[1]
    y_true = np.zeros((n_windows, H, n_items), dtype=np.float32)
    for wi, t in enumerate(window_starts):
        y_true[wi] = D[t:t + H]
    return y_true

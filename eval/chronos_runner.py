"""
Chronos zero-shot rolling-origin inference.

Loads a Chronos pipeline from HuggingFace (uses HF_HOME cache so no
network is needed once the checkpoint is local), runs L=512 → H=30
inference per channel batched across channels, and reduces the 20
sample paths to the per-day median for point-forecast metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path
import time

import numpy as np
import torch

from chronos import ChronosPipeline


def load_chronos(model_id: str, device: str | None = None,
                 dtype: torch.dtype | None = None) -> ChronosPipeline:
    """Load Chronos pipeline; auto-detect device/dtype if not given."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = (torch.bfloat16 if device == "cuda"
                 else torch.float32)
    print(f"  loading {model_id} on {device} ({dtype})", file=sys.stderr)
    return ChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype,
    )


def predict_rolling_origin(
    pipe: ChronosPipeline,
    D: np.ndarray,                # (T, n_items)
    window_starts: list[int],     # rolling-origin t values; forecast [t, t+H)
    L: int = 512, H: int = 30,
    num_samples: int = 20,
    channel_batch: int = 16,
) -> np.ndarray:
    """Returns y_pred of shape (n_windows, H, n_items), point=median."""
    n_windows = len(window_starts)
    n_items = D.shape[1]
    y_pred = np.zeros((n_windows, H, n_items), dtype=np.float32)

    t0 = time.time()
    for wi, t in enumerate(window_starts):
        ctx = D[t - L:t, :]                           # (L, n_items)
        # Predict all channels for this window in batches.
        for j0 in range(0, n_items, channel_batch):
            j1 = min(j0 + channel_batch, n_items)
            # ChronosPipeline.predict expects a list of 1-D tensors.
            ctxs = [torch.tensor(ctx[:, j], dtype=torch.float32)
                    for j in range(j0, j1)]
            samples = pipe.predict(
                ctxs, prediction_length=H,
                num_samples=num_samples, limit_prediction_length=False,
            )
            # samples: (batch, num_samples, H) -- median over samples
            samples = samples.cpu().to(torch.float32).numpy()
            med = np.median(samples, axis=1)          # (batch, H)
            y_pred[wi, :, j0:j1] = med.T              # (H, batch)
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

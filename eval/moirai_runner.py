"""
Moirai zero-shot rolling-origin inference.

Wraps Salesforce moirai (uni2ts.model.moirai) to forecast our supply-chain
demand release. Multivariate-native: a single forward predicts all C items.

Designed to mirror chronos_runner.py's I/O contract so metrics.py can stay
unchanged: produces y_pred of shape (n_windows, H, n_items) with point
forecast = median over num_samples draws.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split as gluonts_split

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


def load_moirai(model_id: str, prediction_length: int, context_length: int,
                target_dim: int, num_samples: int = 100,
                patch_size: int | str = 32, batch_size: int = 32,
                device: str | None = None):
    """Build a Moirai predictor from a pretrained checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading {model_id}  on {device}  "
          f"(L={context_length}, H={prediction_length}, "
          f"target_dim={target_dim}, patch_size={patch_size}, "
          f"num_samples={num_samples})", file=sys.stderr)
    module = MoiraiModule.from_pretrained(model_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size)
    return predictor


def build_test_data(D: np.ndarray, item_ids: list[str],
                    val_end: int, H: int, stride: int,
                    max_windows: int | None = None):
    """Build a GluonTS multivariate test dataset for rolling-origin eval.

    The test region is [val_end, T). Window i has its forecast horizon at
    [val_end + i*stride, val_end + i*stride + H), with full-history input
    (the model itself trims to the last `context_length` steps).
    """
    T, n_items = D.shape
    test_len = T - val_end

    # Wide DataFrame: each column is one item; daily frequency.
    df = pd.DataFrame(
        D,
        columns=item_ids,
        index=pd.date_range("2000-01-01", periods=T, freq="D"),
    )
    ds = PandasDataset(dict(df))
    grouper = MultivariateGrouper(len(ds))
    multivar_ds = grouper(ds)

    train, test_template = gluonts_split(multivar_ds, offset=-test_len)

    n_windows = (test_len - H) // stride + 1
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    test_data = test_template.generate_instances(
        prediction_length=H,
        windows=n_windows,
        distance=stride,
    )
    return test_data, n_windows


def predict_rolling_origin(predictor, test_data, n_windows: int,
                           H: int, n_items: int) -> np.ndarray:
    """Returns y_pred of shape (n_windows, H, n_items); point=median samples."""
    y_pred = np.zeros((n_windows, H, n_items), dtype=np.float32)
    t0 = time.time()
    forecasts = predictor.predict(test_data.input)
    for wi, fc in enumerate(forecasts):
        # fc.samples shape: (num_samples, H, target_dim)
        samples = np.asarray(fc.samples, dtype=np.float32)
        med = np.median(samples, axis=0)          # (H, target_dim)
        y_pred[wi] = med
        if wi == 0 or (wi + 1) % 10 == 0 or wi + 1 == n_windows:
            elapsed = time.time() - t0
            rate = (wi + 1) / max(elapsed, 1e-9)
            eta = (n_windows - wi - 1) / max(rate, 1e-9)
            print(f"  window {wi+1:4d}/{n_windows}  "
                  f"elapsed={elapsed:6.1f}s  "
                  f"rate={rate:5.2f} win/s  "
                  f"eta={eta/60:5.1f}min", file=sys.stderr)
    return y_pred


def collect_y_true(test_data, n_windows: int, H: int,
                   n_items: int) -> np.ndarray:
    """Extract ground truth labels: shape (n_windows, H, n_items)."""
    y_true = np.zeros((n_windows, H, n_items), dtype=np.float32)
    for wi, lbl in enumerate(test_data.label):
        # lbl["target"] shape: (target_dim, H)
        y_true[wi] = np.asarray(lbl["target"], dtype=np.float32).T
    return y_true

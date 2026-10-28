"""Synthetic metric sensitivity study for the ADoTA metrics.

This script compares the behavior of the paper's metric functions under simple
error modes: perfect prediction, additive bias, multiplicative scaling, spatial
shift, and a localized hotspot error.

The goal is not to benchmark the model, but to understand each metric's blind
spots and sensitivities.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.adota_metrics.src.adota.config import DEFAULT_GAMMA_PARAMS
from scripts.adota_metrics.src.metrics.classic import (
    calculate_pure_mape,
    calculate_relative_dose_error,
    calculate_rmse,
)
from scripts.adota_metrics.src.metrics.gamma_pass_rate import gamma_index


def configure_runtime() -> None:
    """Keep the numeric backends stable and avoid thread explosion."""
    n_threads = str(min(4, os.cpu_count() or 1))
    os.environ.setdefault("OMP_NUM_THREADS", n_threads)
    os.environ.setdefault("NUMBA_NUM_THREADS", n_threads)


def safe_gamma(ref: np.ndarray, pred: np.ndarray, resolution=(2.0, 2.0, 2.0)):
    """Wrap the gamma pass-rate call and guard the exact-perfect edge case.

    The ADoTA gamma implementation can raise ZeroDivisionError for an exactly
    perfect prediction because no voxel has gamma > 0; in that case we report a
    missing first pass rate and set the standard pass rate to 1.0.
    """
    scale = {"y_min": 0.0, "y_max": float(ref.max())}
    try:
        _, rates = gamma_index(
            ref.copy(),
            pred.copy(),
            scale,
            DEFAULT_GAMMA_PARAMS,
            resolution,
        )
        return float(rates[0]), float(rates[1])
    except ZeroDivisionError:
        return np.nan, 1.0


def metric_row(ref: np.ndarray, pred: np.ndarray) -> dict:
    """Compute the scalar ADoTA metric values for one prediction."""
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction must share the same shape.")

    mask = ref > 0.10 * ref.max()

    rmse = calculate_rmse(pred, ref)
    mape_all = calculate_pure_mape(pred, ref)
    mape_10 = calculate_pure_mape(pred[mask], ref[mask])
    rho = calculate_relative_dose_error(pred, ref)
    gamma0, gamma1 = safe_gamma(ref, pred)

    return {
        "RMSE": rmse,
        "MAPE_all": mape_all,
        "MAPE_10pct": mape_10,
        "rho": rho,
        "gamma_pass_rate_0": gamma0,
        "gamma_pass_rate_1": gamma1,
    }


def build_cases(ref: np.ndarray) -> dict[str, np.ndarray]:
    """Generate controlled synthetic prediction errors."""
    cases = {
        "perfect": ref.copy(),
        "global_bias": ref + 0.20,
        "global_scale_up": ref * 1.20,
        "global_scale_down": ref * 0.80,
        "spatial_shift_x": np.roll(ref, 1, axis=0),
        "hotspot_drop": ref.copy(),
    }

    cases["hotspot_drop"][10:15, 5:10, 5:10] *= 0.50
    cases["hotspot_drop"] = np.clip(cases["hotspot_drop"], 0.0, None)

    return cases


def main() -> pd.DataFrame:
    """Run the synthetic sensitivity experiment and print a summary table."""
    configure_runtime()

    rng = np.random.default_rng(0)
    ref = rng.random((48, 16, 16), dtype=np.float64)

    rows = []
    for name, pred in build_cases(ref).items():
        row = metric_row(ref, pred)
        row["case"] = name
        rows.append(row)

    df = pd.DataFrame(rows).set_index("case")
    print(df.round(4).to_string())
    return df


if __name__ == "__main__":
    main()

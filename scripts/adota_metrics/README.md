# ADoTA metrics, quick start

The metric code of the ADoTA repository, copied unchanged, plus one notebook that evaluates every metric of the paper once on random input, to show how the functions are initialised and called.

- Upstream: https://github.com/Medical-Physics-Technology/adota (MIT), commit `050e5897bb55422a3c5481bd87c3c19e06469af9` (v1.5.0)
- Paper: Stryja, Lathouwers, Perkó, *Angle dependent dose transformer algorithm for fast proton therapy dose calculations*, arXiv:2602.04375

## Contents

| File | What it is |
|---|---|
| `adota_metrics_quickstart.ipynb` | the notebook, executed (outputs included); a re-run takes about 15 s |
| `src/` | 9 files copied byte for byte from the repository, see the table below; nothing was modified or re-implemented |

## Which paper metric is computed where

| Metric in the paper | Function | File under `src/` |
|---|---|---|
| gamma pass rate, $\Gamma$(dose %, DTA mm, cutoff %): Tables 1, 4, 5, Figs 4, 11, 12 | `gamma_index`, `gamma_index_torch`; `plan_gamma` runs several criteria | `metrics/gamma_pass_rate.py`, `metrics/plan_gamma.py` |
| $\rho$, relative dose error: Eq. 7, Table 2 | `calculate_relative_dose_error` | `metrics/classic.py` |
| RMSE: Table 2 | `calculate_rmse` (units: `to_gy`) | `metrics/classic.py`, `utils/unit_conversions.py` |
| MAPE over voxels above 10 % of the reference maximum: Eq. 8, Figs 5-10 | `calculate_pure_mape` (the mask is built by the caller) | `metrics/classic.py` |

`adota/config.py` is only there because `plan_gamma.py` and the notebook take `DEFAULT_GAMMA_PARAMS` from it; the `__init__.py` files are the upstream package markers.
The notebook ends with a short list of things to keep in mind: argument orders, the two pass-rate definitions and the RMSE unit.
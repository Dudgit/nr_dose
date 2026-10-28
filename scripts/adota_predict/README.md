# ADoTA predict, quick start

The files of the ADoTA repository that are needed to load a trained ADoTA network and call it on beamlet inputs, copied unchanged, plus one notebook that runs a forward pass on a synthetic beamlet to show how the network is loaded, called and read out, and that checks the published network against textbook proton physics. No data, no training code, no evaluation scripts, no weights.

- Upstream: https://github.com/Medical-Physics-Technology/adota (MIT), commit `050e5897bb55422a3c5481bd87c3c19e06469af9` (v1.5.0)
- Paper: Stryja, Lathouwers, Perkó, *Angle dependent dose transformer algorithm for fast proton therapy dose calculations*, arXiv:2602.04375

## Contents

| File | What it is |
|---|---|
| `adota_predict_quickstart.ipynb` | the notebook, executed with the published weights `DoTA_v3_grid_search_v11` (outputs included; the weights are not in this folder); without weights the same cells run with a randomly initialised network and section 8 is skipped; a re-run takes about 40 s on 2 CPU cores (about 10 s without weights) |
| `src/` | 14 files (about 1,900 lines) copied byte for byte from the repository, see the table below; nothing was modified or re-implemented |
| `models/` | only a `README.md`; the trained model folder goes here, see there for where to get it |

One beamlet goes in: `ct_crop` (HU) and `flux_crop`, both `(60, 60, 320)` at 1 mm with the depth axis last, and the beam energy in MeV. One dose comes out: the dose of that pencil beam on the same grid, in eV/g per proton. The notebook walks through it, checks the published network in section 8 and ends with a short list of things to keep in mind (normalisation constants, axis order, energy range, flux width, unit, where the inputs come from).

## What is called where

| Step | Function | File under `src/` |
|---|---|---|
| the network | `DoTA3D_v3` | `adota/models.py`; its blocks are in `adota/layers/` (`conv.py`, `encoder_decoder.py`, `transformer.py`, `tensor_ops.py`) |
| load a model folder (`best_model.pth` + `hyperparams.json`) | `load_model` | `adota/utils.py` |
| beamlet arrays to model input | `prepare_input_from_arrays` (one beamlet), `prepare_inputs_from_arrays_batched` (many) | `loaders/dir_based.py` |
| the same from `<id>_ct.npy`, `<id>_flux.npy`, `<id>_sim_res.json` files | `get_single_record_no_gt` | `loaders/dir_based.py` |
| model output to dose | `postprocess_prediction` (one), `postprocess_predictions_batched` (many) | `loaders/dir_based.py`; `inverse_minmax` in `utils/scallers.py` |
| normalisation constants, device choice | `DEFAULT_SCALE`, `get_device` | `adota/config.py` |
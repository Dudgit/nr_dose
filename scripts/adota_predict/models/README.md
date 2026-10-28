# models/

Put one folder per trained ADoTA model here. A model folder holds two files (this is the layout of the ADoTA repository); the published ones also carry a short note from the authors:

```
models/
└── DoTA_v3_grid_search_v11/
    ├── best_model.pth        PyTorch weights (a state dict)
    ├── hyperparams.json      architecture settings
    └── comments.txt          the authors' note on the training run (optional, the notebook prints it)
```

The authors publish pre-trained models on Google Drive (the link is in `models/README.md` of the ADoTA repository):

https://drive.google.com/drive/folders/1gf5antKnnS7x8QvCgAiPjgF_OizWBxtt?usp=sharing

Download a model folder (for example `DoTA_v3_grid_search_v11`) and copy it here; a Google Drive download arrives as a zip with the model folder inside, which you unzip inside `models/`. The notebook then uses it: the first folder here that holds both files, or the one named in `MODEL_NAME`.

## The model tested with this notebook

`DoTA_v3_grid_search_v11`: `best_model.pth` is 59,752,514 bytes with SHA-256 `c4a1e8190edccb76e92302d2bb0e937008914692ec3179a63065023575f3324b`, 14,927,651 parameters. Its `hyperparams.json` (arrays shown on one line):

```json
{
    "input_shape": [2, 160, 30, 30],
    "output_shape": [1, 160, 30, 30],
    "zero_padding": true,
    "last_activation": false,
    "num_levels": 4,
    "enc_features": 32,
    "kernel_size": 3,
    "convolutional_steps": 1,
    "conv_hidden_channels": 64,
    "num_transformers": 1,
    "num_heads": 4,
    "dropout_rate": 0.1,
    "causal": true,
    "num_forward": 2
}
```

Its `comments.txt` says that it was trained on the full 71k set (a lungs and pelvis dataset, 20 % validation split) and that no GPR was calculated.

Without a model folder the notebook builds a randomly initialised network from the example `hyperparams.json` of the ADoTA repository, so that it runs anywhere (shapes and units only):

```json
{
    "input_shape": [2, 160, 30, 30],
    "num_levels": 4,
    "enc_features": 32,
    "kernel_size": 3,
    "convolutional_steps": 1,
    "conv_hidden_channels": 64,
    "num_transformers": 1,
    "num_heads": 8,
    "dropout_rate": 0.1,
    "causal": true,
    "num_forward": 0
}
```

Things to know:

- The normalisation constants are not stored in these files. The notebook uses `DEFAULT_SCALE` from `src/adota/config.py`, the scale that the ADoTA repository's scripts pass to the model. For `DoTA_v3_grid_search_v11` its energy range (70-270 MeV) is confirmed by the physics check in section 8 of the notebook, and the different fallback of `loaders/dir_based.py` is not; the dose and CT constants of the two scales differ by 2 % and 0.2 %, which that check cannot resolve.
- The network reads the width of the flux channel as well as the energy (section 8). Build `flux_crop` with the spot sizes of the beam data library the model was trained with; that library is not part of the repository.
- `load_model` calls `torch.load(..., weights_only=False)`, which can run code stored in a pickle file. Load only checkpoints from a source you trust. To check that a file is a plain state dict, `torch.load(path, map_location="cpu", weights_only=True)` refuses everything except tensors and containers; the `best_model.pth` above passes.

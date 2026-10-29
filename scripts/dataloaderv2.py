
import json
from monai.transforms import MapTransform, ResizeWithPadOrCropd , EnsureTyped,Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CenterSpatialCropd
import nibabel as nib
import numpy as np
from monai.data import PersistentDataset, DataLoader, Dataset
import torch
from omegaconf import OmegaConf
import os
from pathlib import Path

def get_loaders(cfg):
    train_list = json.load(open(cfg['train_list'], "r"))
    val_list = json.load(open(cfg['val_list'], "r"))

    train_transforms = Compose([
    LoadImaged(keys=["ct", "gt_dose"]),
    EnsureChannelFirstd(keys=["ct", "gt_dose"]),
    ScaleIntensityRanged(keys=['ct'], a_min=cfg['ct_min'], a_max=cfg['ct_max'], b_min=0.0, b_max=1.0, clip=True),
    #Spacingd(keys=["ct", "gt_dose"], pixdim=cfg['pixdim'], mode='trilinear'),
    #ResizeWithPadOrCropd(keys=["ct", "gt_dose"], spatial_size=cfg['roi_size'])])
    # We neeed the function that cuts out the corresponding slices, implemented in db.ipynb
    
    ])
    train_ds =  Dataset(data=train_list, transform=train_transforms)
    val_ds =  Dataset(data=val_list, transform=train_transforms)
    train_loader = DataLoader(train_ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],persistent_workers=cfg['persistent_workers'],prefetch_factor=cfg['prefetch_factor'],pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],persistent_workers=cfg['persistent_workers'],prefetch_factor=cfg['prefetch_factor'],pin_memory=True)


    return train_loader, val_loader



#### Data loader
"""MONAI data loading pipeline for proton dose prediction."""

import json
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from monai.data import (
    DataLoader,
    Dataset,
    MetaTensor,
    PersistentDataset,
)
from monai.data.utils import pickle_hashing
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    MapTransform,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    SelectItemsd,
    Spacingd,
)
from scripts.metaembedder import (
    InjectEnergyDepositionFieldd,
    InjectGaussianBeamPriord,
    Ray_Info,
)


def scale_dose_by_1000(dose):
    """Scale the dose values to match the reference pipeline."""
    return dose * 1000.0


class ExtractSlabsAroundZ(MapTransform):
    """Crop a fixed number of Z slices around the physical ray source."""

    def __init__(
        self,
        keys,
        source_key="ray_source",
        slice_radius=15,
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.slice_radius = slice_radius

    def __call__(self, data):
        d = dict(data)
        physical_point = np.asarray(d[self.source_key][:3], dtype=np.float64)

        for key in self.key_iterator(d):
            image = d[key]

            # Convert the physical ray source to voxel coordinates.
            affine = image.affine.detach().cpu().numpy()
            inverse_affine = np.linalg.inv(affine)
            voxel_coords = nib.affines.apply_affine(
                inverse_affine,
                physical_point,
            )
            z_index = int(np.round(voxel_coords[2]))

            # Python slicing excludes the end index, hence the +1.
            z_start = z_index - self.slice_radius
            z_end = z_index + self.slice_radius + 1

            z_min = max(0, z_start)
            z_max = min(image.shape[3], z_end)
            cropped = image[:, :, :, z_min:z_max]

            # Pad with zero slices if the crop reaches beyond the image.
            pad_left = max(0, -z_start)
            pad_right = max(0, z_end - image.shape[3])

            if pad_left > 0 or pad_right > 0:
                pad_args = (
                    pad_left, pad_right,
                    0, 0,
                    0, 0,
                )
                padded = F.pad(
                    cropped.as_tensor(),
                    pad_args,
                    mode="constant",
                    value=0,
                )

                # Shift the affine origin to account for left-side padding.
                padded_affine = cropped.affine.detach().cpu().numpy().copy()
                padded_affine[:3, 3] -= (
                    padded_affine[:3, 2] * pad_left
                )

                d[key] = MetaTensor(
                    padded,
                    affine=torch.as_tensor(padded_affine),
                    meta=dict(cropped.meta),
                )
            else:
                d[key] = cropped

        return d


class SaveCTAffined(MapTransform):
    """Store the CT affine separately so it survives persistent caching."""

    def __init__(self, keys=("ct",), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            affine = d[key].affine.detach().cpu().numpy().copy()
            d[f"{key}_affine"] = torch.tensor(
                affine,
                dtype=torch.float64,
            )

        return d


class RestoreCTAffined(MapTransform):
    """Restore CT metadata after reading a cached plain tensor."""

    def __init__(self, keys=("ct",), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            if not isinstance(d[key], MetaTensor):
                affine = d[f"{key}_affine"].detach().cpu()
                d[key] = MetaTensor(d[key], affine=affine)

        return d


class RandChannelDropoutd(MapTransform):
    """Randomly zero out prior channels, matching the reference pipeline."""

    def __init__(self, keys, prob=0.15, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            if random.random() < self.prob:
                d[key] = torch.zeros_like(d[key])

        return d


class PatientIDWrapper(torch.utils.data.Dataset):
    """Add patient and dose identifiers to validation examples."""

    def __init__(self, monai_dataset, raw_json_list):
        self.monai_dataset = monai_dataset
        self.raw_json_list = raw_json_list
        self.transform = getattr(monai_dataset, "transform", None)

    def __len__(self):
        return len(self.monai_dataset)

    def __getitem__(self, idx):
        data = self.monai_dataset[idx]

        ct_path = self.raw_json_list[idx]["ct"]
        dose_path = self.raw_json_list[idx]["gt_dose"]

        data["patient_id"] = ct_path.split("/")[-3]
        data["gt_id"] = dose_path.split("/")[-1]
        return data


def get_post_transforms(cfg):
    """Create the prior-generation and final-selection transforms."""

    transforms = [RestoreCTAffined(keys=["ct"])]
    final_keys = ["ct", "gt_dose", "condition"]

    if cfg.get("inject_geometric_prior", False):
        transforms.append(
            InjectGaussianBeamPriord(
                keys=["geometric_prior"],
                source_key="ray_source",
                target_key="ray_target",
                ref_key="ct",
                sigma=cfg["sigma"],
                flip_lps_to_ras=True,
                prior_mode=cfg["prior_mode"],
            )
        )
        final_keys.append("geometric_prior")

    if cfg.get("inject_energy_field", False):
        transforms.append(
            InjectEnergyDepositionFieldd(
                keys=["field"],
                spacing=cfg["pixdim"],
            )
        )
        final_keys.append("field")

    if cfg.get("drop_priors", False):
        prior_keys = []

        if cfg.get("inject_geometric_prior", False):
            prior_keys.append("geometric_prior")
        if cfg.get("inject_energy_field", False):
            prior_keys.append("field")

        if prior_keys:
            transforms.append(
                RandChannelDropoutd(keys=prior_keys, prob=0.15)
            )

    if cfg.get("ray_info", False):
        transforms.append(
            Ray_Info(keys=["ray_source", "ray_target"])
        )
        final_keys.extend(
            [
                "positions",
                "ray_z_positions",
                "ray_start_anchor",
                "ray_end_offset",
                "ray_bragg_offset",
            ]
        )

    transforms.extend(
        [
            EnsureTyped(
                keys=final_keys,
                track_meta=True,
                dtype=torch.float32,
            ),
            SelectItemsd(keys=final_keys),
        ]
    )

    return Compose(transforms)


def get_loaders(cfg, hw="atlasz"):
    """
    Build the training and validation loaders.

    Pass the selected hardware/profile section, such as cfg["atlasz"],
    rather than the entire YAML configuration.
    """

    with open(cfg["train_list"], "r") as f:
        train_list = json.load(f)

    with open(cfg["val_list"], "r") as f:
        val_list = json.load(f)

    train_transforms = Compose(
        [
            LoadImaged(keys=["ct", "gt_dose"]),
            EnsureChannelFirstd(keys=["ct", "gt_dose"]),
            ScaleIntensityRanged(
                keys=["ct"],
                a_min=cfg["ct_min"],
                a_max=cfg["ct_max"],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Lambdad(
                keys=["gt_dose"],
                func=scale_dose_by_1000,
            ),
            ExtractSlabsAroundZ(
                keys=["ct", "gt_dose"],
                source_key="ray_source",
                slice_radius=15,
            ),
            Spacingd(
                keys=["ct", "gt_dose"],
                pixdim=cfg["pixdim"],
                mode="trilinear",
            ),
            ResizeWithPadOrCropd(
                keys=["ct", "gt_dose"],
                spatial_size=cfg["roi_size"],
            ),
            SaveCTAffined(keys=["ct"]),
        ]
    )

    post_transforms = get_post_transforms(cfg)
    dataset_type = cfg["dataset_type"]

    if dataset_type == "persistent":
        cache_dir = cfg["cache_dir"]

        # Match the reference's Komondor cache location when requested.
        if hw == "komondor":
            scratch_dir = os.environ.get("REAL_SCRATCH")
            if not scratch_dir:
                raise RuntimeError(
                    "REAL_SCRATCH must be set when hw='komondor'."
                )
            cache_dir = os.path.join(scratch_dir, "cache")

        os.makedirs(cache_dir, exist_ok=True)

        train_cached_ds = PersistentDataset(
            data=train_list,
            transform=train_transforms,
            cache_dir=cache_dir,
            hash_transform=pickle_hashing,
        )
        val_cached_ds = PersistentDataset(
            data=val_list,
            transform=train_transforms,
            cache_dir=cache_dir,
            hash_transform=pickle_hashing,
        )

        train_ds = Dataset(
            data=train_cached_ds,
            transform=post_transforms,
        )
        val_ds = Dataset(
            data=val_cached_ds,
            transform=post_transforms,
        )

    elif dataset_type == "dataset":
        all_transforms = Compose(
            [train_transforms, post_transforms]
        )
        train_ds = Dataset(
            data=train_list,
            transform=all_transforms,
        )
        val_ds = Dataset(
            data=val_list,
            transform=all_transforms,
        )

    else:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}"
        )

    val_ds = PatientIDWrapper(val_ds, val_list)

    worker_count = int(cfg["num_workers"])
    loader_options = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": worker_count,
        "pin_memory": True,
    }

    if worker_count > 0:
        loader_options["persistent_workers"] = bool(
            cfg.get("persistent_workers", False)
        )
        if cfg.get("prefetch_factor") is not None:
            loader_options["prefetch_factor"] = int(
                cfg["prefetch_factor"]
            )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_options,
    )

    return train_loader, val_loader
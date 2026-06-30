import json
from monai.transforms import MapTransform, ResizeWithPadOrCropd , EnsureTyped,Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CenterSpatialCropd, ConvertToArrayd
import nibabel as nib
import numpy as np
from monai.data import PersistentDataset, DataLoader, Dataset
import torch
from omegaconf import OmegaConf
import os

class ExtractSlabAroundZ(MapTransform):
    def __init__(self, keys, source_key="condition", margin_slices=15, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.margin_slices = margin_slices

    def __call__(self, data):
        d = dict(data)
        ray_source_phys = d[self.source_key][:3] # [x, y, z]
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # Extract the 4x4 affine matrix from the MONAI MetaTensor
            affine = img.affine.cpu().numpy()
            
            # Invert the affine to map physical coordinates back to voxel coordinates
            inv_affine = np.linalg.inv(affine)
            voxel_coords = nib.affines.apply_affine(inv_affine, ray_source_phys)
            
            # Assuming standard spatial arrangement where Z is the 3rd spatial dimension (idx 2)
            # Shape is [C, H, W, Z] after EnsureChannelFirstd
            z_voxel = int(np.round(voxel_coords[2]))
            
            z_min = max(0, z_voxel - self.margin_slices)
            z_max = min(img.shape[3], z_voxel + self.margin_slices + 1)
            
            # Slice the tensor. MONAI MetaTensors automatically update their affine 
            # metadata under the hood when sliced like this.
            d[key] = img[:, :, :, z_min:z_max]
            
        return d


def get_loaders(hw="atlasz",config_name = "default_config"):

    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    data_cfg = cfg['data']
    hw_cfg = OmegaConf.load(f"configs/hw_config.yaml")

    train_list = json.load(open(data_cfg['train_list'], "r"))
    val_list = json.load(open(data_cfg['val_list'], "r"))

    train_transforms = Compose([
    LoadImaged(keys=["ct", "gt_dose"]),
    EnsureChannelFirstd(keys=["ct", "gt_dose"]),
    # Extract +/- 10 slices based on the physical ray location
    ExtractSlabAroundZ(keys=["ct", "gt_dose"], source_key="condition", margin_slices=data_cfg['margin_slices']),
    # Resample the extracted slabs to a fixed 1x1x1 mm resolution
    Spacingd(keys=["ct", "gt_dose"], pixdim=data_cfg['pixdim'], mode="trilinear"),
    ResizeWithPadOrCropd(keys=["ct", "gt_dose"], spatial_size=data_cfg['roi_size']),
    ConvertToArrayd(keys=["condition"])
    ])

    cache_dir = hw_cfg[hw]['cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    if hw_cfg[hw]["dataset_type"] == "dataset":
        train_ds =  Dataset(data=train_list, transform=train_transforms)
        val_ds =  Dataset(data=val_list, transform=train_transforms)
    else:
        train_ds =  PersistentDataset(data=train_list, transform=train_transforms, cache_dir=cache_dir)
        val_ds =  PersistentDataset(data=val_list, transform=train_transforms, cache_dir=cache_dir)

    train_loader = DataLoader(train_ds,batch_size=hw_cfg[hw]['batch_size'],shuffle=True,num_workers=hw_cfg[hw]['num_workers'])
    val_loader = DataLoader(val_ds,batch_size=hw_cfg[hw]['batch_size'],shuffle=False,num_workers=hw_cfg[hw]['num_workers'])

    return train_loader, val_loader


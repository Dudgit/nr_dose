import json
from monai.transforms import MapTransform, ResizeWithPadOrCropd , EnsureTyped,Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CenterSpatialCropd
import nibabel as nib
import numpy as np
from monai.data import PersistentDataset, DataLoader
import torch
from omegaconf import OmegaConf
import os

class ExtractSlabAroundZ(MapTransform):
    def __init__(self, keys, source_key="ray_source_phys", margin_slices=10, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.margin_slices = margin_slices

    def __call__(self, data):
        d = dict(data)
        ray_source_phys = d[self.source_key] # [x, y, z]
        
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


def get_loaders(batch_size= 32, num_workers = 4,config_name = "default_config"):

    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    data_cfg = cfg['data']
    train_list = json.load(open(data_cfg['train_list'], "r"))
    val_list = json.load(open(data_cfg['val_list'], "r"))

    train_transforms = Compose([
    LoadImaged(keys=["ct", "gt_dose"]),
    EnsureChannelFirstd(keys=["ct", "gt_dose"]),
    
    # Extract +/- 10 slices based on the physical ray location
    ExtractSlabAroundZ(keys=["ct", "gt_dose"], source_key="ray_source_phys", margin_slices=data_cfg['margin_slices']),
    
    # Resample the extracted slabs to a fixed 1x1x1 mm resolution
    Spacingd(keys=["ct", "gt_dose"], pixdim=data_cfg['pixdim'], mode="trilinear"),
    CenterSpatialCropd(keys=["ct", "gt_dose"], roi_size=data_cfg['roi_size']),])

    cache_dir = "/home/nr_dodb/nr_dose_scratch"
    os.makedirs(cache_dir, exist_ok=True)
    train_ds =  PersistentDataset(data=train_list, transform=train_transforms,cache_dir=cache_dir)
    val_ds =  PersistentDataset(data=val_list, transform=train_transforms,cache_dir=cache_dir)

    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader = DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return train_loader, val_loader


from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ResizeWithPadOrCropd,
    EnsureTyped,
    ScaleIntensityRanged)

import torch
from omegaconf import OmegaConf
import os

def get_loader(batch_size= 32, num_workers = 4):

    cfg = OmegaConf.load("config/config.yaml")
    data_cfg = cfg['data']

    transforms = Compose([
        LoadImaged(keys=["ct","dose"]),
        EnsureChannelFirstd(keys=["ct","dose"]),
        Spacingd(keys=["ct","dose"], pixdim=data_cfg['voxel_spacing'],mode ="trilinear"),
        ResizeWithPadOrCropd(keys=["ct","dose"], spatial_size=data_cfg['target_size'],mode ="constant",constant_values=0.0),
        EnsureTyped(keys=["ct","dose"],dtype = torch.float32),       
        ])
    cache_dir = "/home/nr_dodb/nr_dose_scratch"
    os.makedirs(cache_dir, exist_ok=True)
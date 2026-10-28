
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
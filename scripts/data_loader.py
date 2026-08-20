import json
from monai.transforms import MapTransform, ResizeWithPadOrCropd , EnsureTyped,Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CenterSpatialCropd
import nibabel as nib
import numpy as np
from monai.data import PersistentDataset, DataLoader, Dataset
import torch
from omegaconf import OmegaConf
import os
from pathlib import Path

from scripts.metaembedder import InjectGaussianBeamPriord, InjectEnergyDepositionFieldd, Ray_Info

from monai.transforms import ScaleIntensityRanged

from torch.utils.data import Dataset as TorchDataset

class PatientIDWrapper(TorchDataset):
    def __init__(self, monai_dataset, raw_json_list):
        self.monai_dataset = monai_dataset
        self.raw_json_list = raw_json_list

        if hasattr(monai_dataset, "transform"):
            self.transform = monai_dataset.transform
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.monai_dataset)

    def __getitem__(self, idx):
        # 1. Pull the data from the existing 1TB MONAI cache
        data_dict = self.monai_dataset[idx]
        
        # 2. Extract the ID from the original JSON list (bypassing the cache entirely)
        # Example: "/scratch/db/proton/training/1ABB161/image/ct.mha" -> "1ABB161"
        file_path = self.raw_json_list[idx]["ct"]
        patient_id = file_path.split('/')[-3]
        fpath2 = self.raw_json_list[idx]["gt_dose"]
        gt_id = fpath2.split('/')[-1]
        
        # 3. Inject the string into the dictionary and return
        data_dict["patient_id"] = patient_id
        data_dict["gt_id"] = gt_id
        return data_dict



import numpy as np
import nibabel as nib
from monai.transforms import MapTransform
from monai.data import MetaTensor

class ExtractSlabsAroundZ(MapTransform):
    def __init__(self, keys, source_key="ray_source", slice_radius=15, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        # 15 slices above + 15 slices below + 1 center slice = 31 slices total
        self.slice_radius = slice_radius

    def __call__(self, data):
        d = dict(data)
        
        # 1. Extract the physical point (X, Y, Z)
        physical_point = np.array(d[self.source_key][:3])
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # 2. Get the affine matrix from this specific image
            
            affine = img.affine.cpu().numpy()
            #d['affine_trans'] = affine
            # 3. Invert the affine to create a "Physical to Voxel" mapping
            inv_affine = np.linalg.inv(affine)
            
            # 4. Apply the inverted affine to your physical point
            voxel_coords = nib.affines.apply_affine(inv_affine, physical_point)
            
            # 5. Extract the Z index (the 3rd element)
            z_index = int(np.round(voxel_coords)[2])
            
            # 6. Calculate theoretical Z boundaries (can exceed image shape)
            # We add +1 to z_end because Python slicing is exclusive at the end
            z_start = z_index - self.slice_radius
            z_end = z_index + self.slice_radius + 2
            
            # 7. Clamp bounds for safe numpy slicing
            z_min = max(0, z_start)
            z_max = min(img.shape[3], z_end)
            
            # 8. Crop the slab (Assuming shape is [Channel, X, Y, Z])
            # Notice we keep all of X and Y intact!
            cropped_img = img[:, :, :, z_min:z_max].clone()
            
            # 9. Calculate exactly where Z-padding belongs if we hit a boundary
            pad_z_left = max(0, -z_start)
            pad_z_right = max(0, z_end - img.shape[3])
            
            # 10. Apply PyTorch Pad (F.pad goes from the last dim backwards)
            # We pass 0s for Y and X so ONLY the Z axis gets padded.
            if pad_z_left > 0 or pad_z_right > 0:
                cropped_img = torch.nn.functional.pad(
                    torch.as_tensor(cropped_img), 
                    (pad_z_left, pad_z_right, 0, 0, 0, 0), # (Z_left, Z_right, Y_left, Y_right, X_left, X_right)
                    mode='constant', value=0
                )
            d['orig_shape'] = np.array(img.shape[1:3], dtype=np.float32)
            d[key] = cropped_img
            d['affine_trans'] = affine
            d['z_start'] = np.array([z_start], dtype=np.float32)
            
        return d


import random
class RandChannelDropoutd(MapTransform):
    """
    Randomly zeroes out entire image channels with a given probability
    to prevent the model from over-relying on spatial priors.
    """
    def __init__(self, keys, prob=0.15, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if random.random() < self.prob:
                # Completely black out this prior for the current forward pass
                d[key] = torch.zeros_like(d[key])
        return d

from monai.transforms import Lambdad
from monai.transforms import SelectItemsd

class BackupFinalAffined(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # Save the fully updated affine as a standard array
            if hasattr(img, "affine"):
                d[f"{key}_affine_trans"] = img.affine.cpu().numpy()
            elif hasattr(img, "meta") and "affine" in img.meta:
                d[f"{key}_affine_trans"] = img.meta["affine"].cpu().numpy()
        return d


def scale_dose_by_1000(x):
    return x * 1000.0

from monai.transforms import ScaleIntensityd

def get_post_transforms(cfg):
    dynamic_post_transform_list = []
    final_keys = ["ct", "gt_dose", "condition"]
    if cfg['inject_geometric_prior']:
        #dynamic_post_transform_list.append(InjectGaussianBeamPriord_v2(keys=['geometric_prior'], ref_key="ct", sigma=cfg['sigma'],flip_lps_to_ras=False))
        final_keys.append("ray_source")
        final_keys.append("ray_target")
        final_keys.append("geometric_prior")
    if cfg['inject_energy_field']:
        dynamic_post_transform_list.append(InjectEnergyDepositionFieldd(keys=['field'], spacing=cfg['pixdim']))
        final_keys.append("field")
    if cfg['drop_priors']:
        dynamic_post_transform_list.append(RandChannelDropoutd(keys=['geometric_prior', 'field'], prob=0.15))
    if cfg['ray_info']:
        dynamic_post_transform_list.append(Ray_Info(keys=['ray_source', 'ray_target']))
        final_keys.extend(["positions", "ray_start_anchor", "ray_end_offset", "ray_bragg_offset"])
    
    dynamic_post_transform_list.append(EnsureTyped(keys=final_keys, track_meta=False, dtype=torch.float32))
    dynamic_post_transform_list.append(SelectItemsd(keys=final_keys))
    return dynamic_post_transform_list
    
def get_loaders(cfg,hw = "atlasz"):
    train_list = json.load(open(cfg['train_list'], "r"))
    val_list = json.load(open(cfg['val_list'], "r"))

    
    train_transforms = Compose([
    LoadImaged(keys=["ct", "gt_dose"]),
    EnsureChannelFirstd(keys=["ct", "gt_dose"]),
    ExtractSlabsAroundZ(keys=["ct", "gt_dose"], source_key="ray_source", slice_radius=15),
    ScaleIntensityd(keys=["gt_dose"], factor=1000.0),
    ScaleIntensityRanged(keys=['ct'], a_min=cfg['ct_min'], a_max=cfg['ct_max'], b_min=0.0, b_max=1.0, clip=True),
    ResizeWithPadOrCropd(keys=["ct", "gt_dose"], spatial_size=cfg['roi_size']),
    #InjectGaussianBeamPriord(keys=['geometric_prior'], source_key="ray_source", target_key="ray_target", ref_key="ct", sigma=cfg['sigma'], flip_lps_to_ras=True, prior_mode=cfg['prior_mode']),
    #InjectEnergyDepositionFieldd(keys=['field'], spacing=cfg['pixdim']),
    EnsureTyped(keys=["ct", "gt_dose","ray_source","ray_target","condition","affine_trans","z_start","orig_shape"], track_meta=False )])




    #dynamic_post_transforms = Compose(get_post_transforms(cfg))
    
    if cfg['dataset_type'] == "persistent":
        cache_dir = cfg['cache_dir']
        if hw == "komondor":
            scratch_dir = os.environ.get("REAL_SCRATCH")
            cache_dir = os.path.join(scratch_dir,"cache")
            print(f"Using cache directory: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)

        train_cached_ds =  PersistentDataset(data=train_list, transform=train_transforms, cache_dir=cache_dir)
        val_cached_ds =  PersistentDataset(data=val_list, transform=train_transforms, cache_dir=cache_dir)
        train_ds = Dataset(data=train_cached_ds, transform=None)
        val_ds = Dataset(data=val_cached_ds, transform=None)

    if cfg["dataset_type"] == "dataset":
        train_ds =  Dataset(data=train_list, transform=train_transforms)
        val_ds =  Dataset(data=val_list, transform=train_transforms)
    
    val_ds = PatientIDWrapper(val_ds, val_list)
    train_loader = DataLoader(train_ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],persistent_workers=cfg['persistent_workers'],prefetch_factor=cfg['prefetch_factor'],pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],persistent_workers=cfg['persistent_workers'],prefetch_factor=cfg['prefetch_factor'],pin_memory=True)

    return train_loader, val_loader


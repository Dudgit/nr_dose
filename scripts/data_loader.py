import json
from monai.transforms import MapTransform, ResizeWithPadOrCropd , EnsureTyped,Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CenterSpatialCropd
import nibabel as nib
import numpy as np
from monai.data import PersistentDataset, DataLoader, Dataset
import torch
from omegaconf import OmegaConf
import os
from pathlib import Path

from scripts.metaembedder import InjectGaussianBeamPriord, InjectEnergyDepositionFieldd

from monai.transforms import ScaleIntensityRanged

from torch.utils.data import Dataset as TorchDataset

class PatientIDWrapper(TorchDataset):
    def __init__(self, monai_dataset, raw_json_list):
        self.monai_dataset = monai_dataset
        self.raw_json_list = raw_json_list

    def __len__(self):
        return len(self.monai_dataset)

    def __getitem__(self, idx):
        # 1. Pull the data from the existing 1TB MONAI cache
        data_dict = self.monai_dataset[idx]
        
        # 2. Extract the ID from the original JSON list (bypassing the cache entirely)
        # Example: "/scratch/db/proton/training/1ABB161/image/ct.mha" -> "1ABB161"
        file_path = self.raw_json_list[idx]["ct"]
        patient_id = file_path.split('/')[-3]
        
        # 3. Inject the string into the dictionary and return
        data_dict["patient_id"] = patient_id
        return data_dict



import numpy as np
import nibabel as nib
from monai.transforms import MapTransform

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
            
            # 3. Invert the affine to create a "Physical to Voxel" mapping
            inv_affine = np.linalg.inv(affine)
            
            # 4. Apply the inverted affine to your physical point
            voxel_coords = nib.affines.apply_affine(inv_affine, physical_point)
            
            # 5. Extract the Z index (the 3rd element)
            z_index = int(np.round(voxel_coords)[2])
            
            # 6. Calculate theoretical Z boundaries (can exceed image shape)
            # We add +1 to z_end because Python slicing is exclusive at the end
            z_start = z_index - self.slice_radius
            z_end = z_index + self.slice_radius + 1 
            
            # 7. Clamp bounds for safe numpy slicing
            z_min = max(0, z_start)
            z_max = min(img.shape[3], z_end)
            
            # 8. Crop the slab (Assuming shape is [Channel, X, Y, Z])
            # Notice we keep all of X and Y intact!
            cropped_img = img[:, :, :, z_min:z_max]
            
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
                
            d[key] = cropped_img
            
        return d
    

from monai.transforms import Lambdad
from monai.transforms import SelectItemsd



def scale_dose_by_1000(x):
    return x * 1000.0
def get_loaders(cfg,hw = "atlasz"):
    train_list = json.load(open(cfg['train_list'], "r"))
    val_list = json.load(open(cfg['val_list'], "r"))

    
    train_transforms = Compose([
    LoadImaged(keys=["ct", "gt_dose"]),
    EnsureChannelFirstd(keys=["ct", "gt_dose"]),
    Lambdad(keys=["gt_dose"], func=scale_dose_by_1000),
    ScaleIntensityRanged(keys=['ct'],a_min=cfg['ct_min'], a_max=cfg['ct_max'], b_min=0.0, b_max=1.0, clip=True),
    ExtractSlabsAroundZ(keys=["ct", "gt_dose"], source_key="ray_source", slice_radius=15),
    Spacingd(keys=["ct", "gt_dose"], pixdim=cfg['pixdim'], mode='nearest'), # pixdim = [4.0, 4.0, 3.0]
    ResizeWithPadOrCropd(keys=["ct", "gt_dose"], spatial_size=cfg['roi_size']), #roi_size = [128, 128, 32] 
    InjectGaussianBeamPriord(keys =['ct'],source_key="ray_source", target_key="ray_target", ref_key="ct", sigma=cfg['sigma'], flip_lps_to_ras = True,prior_mode = "full_line"),
    InjectEnergyDepositionFieldd(keys =['field'],spacing=cfg['pixdim']),
    EnsureTyped(keys=["ct", "gt_dose", "condition", "geometric_prior","ray_source", "ray_target","field"],track_meta=False,dtype=torch.float32),
    SelectItemsd(keys=["ct", "gt_dose", "condition", "geometric_prior","ray_source", "ray_target","field"])
    ])
    
    if cfg['dataset_type'] == "persistent":
        cache_dir = cfg['cache_dir']
        if hw == "komondor":
            scratch_dir = os.environ.get("REAL_SCRATCH")
            cache_dir = os.path.join(scratch_dir,"cache")
            print(f"Using cache directory: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        train_ds =  PersistentDataset(data=train_list, transform=train_transforms, cache_dir=cache_dir)
        val_ds =  PersistentDataset(data=val_list, transform=train_transforms, cache_dir=cache_dir)
    if cfg["dataset_type"] == "dataset":
        train_ds =  Dataset(data=train_list, transform=train_transforms)
        val_ds =  Dataset(data=val_list, transform=train_transforms)
    
    val_ds = PatientIDWrapper(val_ds, val_list)
    train_loader = DataLoader(train_ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],persistent_workers=cfg['persistent_workers'],prefetch_factor=cfg['prefetch_factor'],pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],persistent_workers=cfg['persistent_workers'],prefetch_factor=cfg['prefetch_factor'],pin_memory=True)

    return train_loader, val_loader


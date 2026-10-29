import torch
import torch.nn.functional as F
from monai.transforms import MapTransform, LoadImageD, ScaleIntensityRanged, EnsureChannelFirstd, Compose, NormalizeIntensityD, ToTensord
import numpy as np
import scipy.ndimage as ndimage
import json
from monai.data import DataLoader, Dataset, PersistentDataset


class MaskAirDosed(MapTransform):
    def __init__(self, keys=["ct", "dose"], ct_key="ct", dose_key="dose", threshold=-500):
        super().__init__(keys)
        self.ct_key = ct_key
        self.dose_key = dose_key
        self.threshold = threshold

    def __call__(self, data):
        d = dict(data)
        ct_tensor = d[self.ct_key]
        dose_tensor = d[self.dose_key]
        
        ct_data = ct_tensor.cpu().numpy()
        tissue_mask = ct_data > self.threshold
        
        # --- NEW: Largest Connected Component (LCC) Extraction ---
        # Labels all distinct blobs of tissue/dense material
        labeled_mask, num_features = ndimage.label(tissue_mask)
        
        if num_features > 0:
            # Count the size of each blob
            component_sizes = np.bincount(labeled_mask.ravel())
            
            # Set background (label 0) size to zero so we don't accidentally pick the air
            component_sizes[0] = 0
            
            # Find the ID of the largest blob (the patient)
            largest_component_label = component_sizes.argmax()
            
            # Redefine the mask to ONLY include the patient
            tissue_mask = (labeled_mask == largest_component_label)
        
        # Fill the internal cavities (lungs, bowel) inside the isolated patient
        body_mask = ndimage.binary_fill_holes(tissue_mask)
        
        body_mask_tensor = torch.tensor(body_mask, device=dose_tensor.device, dtype=dose_tensor.dtype)
        
        # Zero out the external dose
        d[self.dose_key] = dose_tensor * body_mask_tensor
        
        return d


class BeamEyeViewCropd(MapTransform):
    def __init__(self, keys, crop_size=(512, 32, 32)):
        # crop_size: (Beam Path, Transverse Scatter, Z-Slices)
        super().__init__(keys)
        self.crop_size = crop_size

    def __call__(self, data):
        d = dict(data)
        
        angle_deg = d["angle"]
        angle_rad = np.radians(angle_deg)
        
        for key in self.keys:
            tensor = d[key]  
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor)
                
            C, D_in, H_in, W_in = tensor.shape
            D_out, H_out, W_out = self.crop_size
            
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # THE FIX: Negating the 3rd column (Col 2). 
            # This reverses the read direction, mapping Index 0 to the source location.
            affine_matrix = torch.tensor([
                # Col 0 (32-Slices)   Col 1 (32-Scatter)         Col 2 (Beam Path reversed) Col 3
                [W_out / W_in,        0.0,                       0.0,                       0.0],  # Patient Z
                [0.0,                 (H_out / H_in) * sin_a,   -(D_out / H_in) * cos_a,    0.0],  # Patient Y
                [0.0,                 (H_out / D_in) * cos_a,    (D_out / D_in) * sin_a,    0.0]   # Patient X
            ], dtype=torch.float32).unsqueeze(0)
            
            grid = F.affine_grid(
                affine_matrix, 
                size=(1, C, D_out, H_out, W_out), 
                align_corners=False
            )
            
            tensor_b = tensor.unsqueeze(0).float()
            rotated_crop = F.grid_sample(
                tensor_b, 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            )
            
            d[key] = rotated_crop.squeeze(0)
            
        return d


def get_transforms():
    transforms = Compose([
        LoadImageD(keys=["ct", "dose"],image_only = False),
        ScaleIntensityRanged(keys=["ct"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityD(keys=['dose'],channel_wise=True),
        EnsureChannelFirstd(keys=["ct", "dose"]),
        MaskAirDosed(keys=["ct", "dose"], ct_key="ct", dose_key="dose", threshold=-500),
        BeamEyeViewCropd(keys=["ct", "dose"], crop_size=(512, 32, 32)),
        ToTensord(keys=["ct", "dose","beam_source","beam_target","ct_meta_dict"])

        ])
    return transforms

def create_dataloader(cfg):
    
    transforms = get_transforms()
    
    with open(cfg["data"]['train_list'], "r") as f:
        train_list = json.load(f)
    with open(cfg["data"]['val_list'], "r") as f:
        val_list = json.load(f)
   
    if cfg.get('persistent_dataset', False):
        train_dataset = PersistentDataset(data=train_list, transform=transforms)
        val_dataset = PersistentDataset(data=val_list, transform=transforms)
    else:
        train_dataset = Dataset(data=train_list, transform=transforms)
        val_dataset = Dataset(data=val_list, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=cfg["data"]['batch_size'], shuffle=True, num_workers=cfg["data"]['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg["data"]['batch_size'], shuffle=False, num_workers=cfg["data"]['num_workers'])
   
    return train_loader, val_loader

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/flare_config.yaml")
    train_loader, val_loader = create_dataloader(cfg)
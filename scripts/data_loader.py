import json
from monai.transforms import MapTransform, ResizeWithPadOrCropd , EnsureTyped,Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CenterSpatialCropd
import nibabel as nib
import numpy as np
from monai.data import PersistentDataset, DataLoader, Dataset
import torch
from omegaconf import OmegaConf
import os

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


import numpy as np
import nibabel as nib
from monai.transforms import MapTransform

class ExtractDynamicROId(MapTransform):
    def __init__(self, keys, source_key="condition", roi_size=(64, 64, 32), asymmetric_x_shift=0, allow_missing_keys=False):
        """
        Dynamically crops a 3D ROI around a physical coordinate.
        
        Args:
            roi_size: (X, Y, Z) size of the final cropped tensor in voxels.
            asymmetric_x_shift: Voxel count to shift the box left (- value) or right (+ value)
                                to capture asymmetric beam entrance tracks.
        """
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.roi_size = roi_size
        self.x_shift = asymmetric_x_shift

    def __call__(self, data):
        d = dict(data)
        ray_source_phys = d[self.source_key][:3] # [x, y, z] in mm
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # 1. Translate Physical -> Voxel Space
            affine = img.affine.cpu().numpy()
            inv_affine = np.linalg.inv(affine)
            voxel_coords = nib.affines.apply_affine(inv_affine, ray_source_phys)
            
            x_v = int(np.round(voxel_coords[0]))
            y_v = int(np.round(voxel_coords[1]))
            z_v = int(np.round(voxel_coords[2]))
            
            # 2. Calculate the bounding box boundaries
            # Shape is [C, X, Y, Z] after EnsureChannelFirstd
            rx, ry, rz = self.roi_size
            
            # Apply the horizontal shift for the left-entering beam
            x_center = x_v + self.x_shift
            
            x_min = max(0, x_center - (rx // 2))
            x_max = min(img.shape[1], x_center + (rx // 2) + (rx % 2))
            
            y_min = max(0, y_v - (ry // 2))
            y_max = min(img.shape[2], y_v + (ry // 2) + (ry % 2))
            
            z_min = max(0, z_v - (rz // 2))
            z_max = min(img.shape[3], z_v + (rz // 2) + (rz % 2))
            
            # 3. Crop the tensor
            cropped_img = img[:, x_min:x_max, y_min:y_max, z_min:z_max]
            
            # 4. Failsafe Padding (In case the Bragg peak is right on the patient's skin edge)
            # If the crop hit the edge of the CT scan, the tensor will be smaller than roi_size.
            # We must pad it with zeros so PyTorch DataLoader doesn't crash trying to stack batches.
            pad_x = rx - cropped_img.shape[1]
            pad_y = ry - cropped_img.shape[2]
            pad_z = rz - cropped_img.shape[3]
            
            if pad_x > 0 or pad_y > 0 or pad_z > 0:
                # F.pad format: (left_z, right_z, left_y, right_y, left_x, right_x)
                cropped_img = torch.nn.functional.pad(
                    cropped_img, 
                    (0, pad_z, 0, pad_y, 0, pad_x), 
                    mode='constant', 
                    value=0
                )
                
            d[key] = cropped_img
            
            # Save coordinates so the Level 2 Metric Callback can paste it back onto the 3D Canvas
            if key == "gt_dose": 
                d["x_min"], d["x_max"] = int(x_min), int(x_max)
                d["y_min"], d["y_max"] = int(y_min), int(y_max)
                d["z_min"], d["z_max"] = int(z_min), int(z_max)
            
        return d
    

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
    

class ExtractVectorROId(MapTransform):
    def __init__(self, keys, source_key="ray_source", target_key="ray_target", roi_size=(64, 64, 32), backward_shift_voxels=16, allow_missing_keys=False,cylinder_radius=4.0):
        """
        Crops an ROI anchored to the target, shifted dynamically backward along the beam's trajectory.
        """
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.target_key = target_key
        self.roi_size = roi_size
        self.backward_shift = backward_shift_voxels
        self.cylinder_radius = cylinder_radius

    def __call__(self, data):
        d = dict(data)
        
        target_phys = np.array(d[self.target_key][:3])
        source_phys = np.array(d[self.source_key][:3])
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # 1. Translate Physical -> Voxel Space
            affine = img.affine.cpu().numpy()
            inv_affine = np.linalg.inv(affine)
            
            target_v = nib.affines.apply_affine(inv_affine, target_phys)
            source_v = nib.affines.apply_affine(inv_affine, source_phys)
            
            # 2. Calculate the Beam Direction Vector
            # Vector pointing FROM the source TO the target
            direction_v = target_v - source_v
            
            # Normalize to a unit vector (length of 1)
            norm = np.linalg.norm(direction_v)
            if norm > 0:
                direction_v = direction_v / norm
                
            # 3. Slide the center backwards along the trajectory
            # This works for ANY gantry angle!
            center_v = target_v - (direction_v * self.backward_shift)
            
            x_c = int(np.round(center_v[0]))
            y_c = int(np.round(center_v[1]))
            z_c = int(np.round(center_v[2]))
            
            rx, ry, rz = self.roi_size
            
            x_min = max(0, x_c - (rx // 2))
            x_max = min(img.shape[1], x_c + (rx // 2) + (rx % 2))
            
            y_min = max(0, y_c - (ry // 2))
            y_max = min(img.shape[2], y_c + (ry // 2) + (ry % 2))
            
            z_min = max(0, z_c - (rz // 2))
            z_max = min(img.shape[3], z_c + (rz // 2) + (rz % 2))
            
            # 4. Crop
            cropped_img = img[:, x_min:x_max, y_min:y_max, z_min:z_max]
            
            # 5. Failsafe Padding
            pad_x = rx - cropped_img.shape[1]
            pad_y = ry - cropped_img.shape[2]
            pad_z = rz - cropped_img.shape[3]
            
            if pad_x > 0 or pad_y > 0 or pad_z > 0:
                cropped_img = torch.nn.functional.pad(
                    cropped_img, 
                    (0, pad_z, 0, pad_y, 0, pad_x), 
                    mode='constant', value=0
                )
                
            d[key] = cropped_img
            
            # Save for the Level 2 Metric Canvas
            if key == "gt_dose": 
                d["x_min"], d["x_max"] = np.array(x_min, dtype=np.int32), np.array(x_max, dtype=np.int32)
                d["y_min"], d["y_max"] = np.array(y_min, dtype=np.int32), np.array(y_max, dtype=np.int32)
                d["z_min"], d["z_max"] = np.array(z_min, dtype=np.int32), np.array(z_max, dtype=np.int32)
            if key == "ct":
                # Create a blank 3D grid of zeros matching your ROI size (64x64x32)
                rx, ry, rz = self.roi_size
                z_grid, y_grid, x_grid = np.meshgrid(
                    np.arange(rz), np.arange(ry), np.arange(rx), indexing='ij'
                )
                
                # The target is located at the center of our cropped box, 
                # shifted slightly by backward_shift_voxels.
                # Let's map the 3D line through the local box coordinates.
                local_center = np.array([rx // 2, (ry // 2) + self.backward_shift, rz // 2])
                
                # Calculate distance from every voxel in the box to the beam trajectory line
                # Formula: distance = || (Point - Center) x Direction ||
                points = np.stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()), axis=-1)
                vec_to_center = points - local_center
                
                # Cross product calculates the perpendicular distance to the line
                cross_prod = np.cross(vec_to_center, direction_v)
                distances = np.linalg.norm(cross_prod, axis=-1)
                
                # Create a binary mask: 1.0 if inside the radius, 0.0 if outside
                binary_mask = (distances <= self.cylinder_radius).astype(np.float32)
                binary_mask = binary_mask.reshape(1, rx, ry, rz) # Add channel dim
                
                # Convert to torch tensor
                prior_tensor = torch.from_numpy(binary_mask)
                
                # Concatenate the CT (Channel 0) and the Prior Mask (Channel 1)
                # Ensure cropped_img is a tensor first
                if not isinstance(cropped_img, torch.Tensor):
                    cropped_img = torch.from_numpy(cropped_img)
                    
                # The final input to the UNet becomes a 2-channel 3D volume!
                d[key] = torch.cat([cropped_img, prior_tensor], dim=0)
            
        return d

from monai.transforms import Lambdad
from monai.transforms import SelectItemsd
def scale_dose_by_1000(x):
    return x * 1000.0
def get_loaders(hw="atlasz",config_name = "default_config"):

    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    data_cfg = cfg['data']
    hw_cfg = OmegaConf.load(f"configs/hw_config.yaml")

    train_list = json.load(open(data_cfg['train_list'], "r"))
    val_list = json.load(open(data_cfg['val_list'], "r"))

    
    train_transforms = Compose([
    LoadImaged(keys=["ct", "gt_dose"]),
    EnsureChannelFirstd(keys=["ct", "gt_dose"]),
    Lambdad(keys=["gt_dose"], func=scale_dose_by_1000),
    ScaleIntensityRanged(keys=['ct'],a_min=data_cfg['ct_min'], a_max=data_cfg['ct_max'], b_min=0.0, b_max=1.0, clip=True),
    ExtractSlabsAroundZ(keys=["ct", "gt_dose"], source_key="ray_source", slice_radius=15),
    Spacingd(keys=["ct", "gt_dose"], pixdim=data_cfg['pixdim'], mode='trilinear'), # pixdim = [4.0, 4.0, 3.0]
    ResizeWithPadOrCropd(keys=["ct", "gt_dose"], spatial_size=data_cfg['roi_size']), #roi_size = [128, 128, 32] 
    EnsureTyped(keys=["ct", "gt_dose", "condition"],track_meta=False,dtype=torch.float32),
    SelectItemsd(keys=["ct", "gt_dose", "condition"])
    ])
    
    if hw_cfg[hw]['dataset_type'] == "persistent":
        cache_dir = hw_cfg[hw]['cache_dir']
        os.makedirs(cache_dir, exist_ok=True)
        train_ds =  PersistentDataset(data=train_list, transform=train_transforms, cache_dir=cache_dir)
        val_ds =  PersistentDataset(data=val_list, transform=train_transforms, cache_dir=cache_dir)
    if hw_cfg[hw]["dataset_type"] == "dataset":
        train_ds =  Dataset(data=train_list, transform=train_transforms)
        val_ds =  Dataset(data=val_list, transform=train_transforms)
    
    val_ds = PatientIDWrapper(val_ds, val_list)
    train_loader = DataLoader(train_ds,batch_size=hw_cfg[hw]['batch_size'],shuffle=True,num_workers=hw_cfg[hw]['num_workers'],persistent_workers=hw_cfg[hw]['persistent_workers'],prefetch_factor=hw_cfg[hw]['prefetch_factor'],pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size=hw_cfg[hw]['batch_size'],shuffle=False,num_workers=hw_cfg[hw]['num_workers'],persistent_workers=hw_cfg[hw]['persistent_workers'],prefetch_factor=hw_cfg[hw]['prefetch_factor'],pin_memory=True)

    return train_loader, val_loader


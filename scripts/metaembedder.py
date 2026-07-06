import torch
import numpy as np
import nibabel as nib
from monai.transforms import MapTransform


class InjectGaussianBeamPriord(MapTransform):
    def __init__(self, keys, source_key="ray_source", target_key="ray_target", ref_key="ct", sigma=4.0, flip_lps_to_ras=True):
        super().__init__(keys)
        self.source_key = source_key
        self.target_key = target_key
        self.ref_key = ref_key
        self.sigma = sigma 
        self.flip_lps_to_ras = flip_lps_to_ras

    def __call__(self, data):
        d = dict(data)
        
        # 1. Extract Physical Coordinates (Make a copy so we don't permanently alter the original data)
        phys_source = np.array(d[self.source_key][:3], copy=True)
        phys_target = np.array(d[self.target_key][:3], copy=True)
        
        # 2. Coordinate System Alignment (LPS -> RAS)
        if self.flip_lps_to_ras:
            # Negating X and Y physically mirrors the points to match MONAI's default RAS space
            phys_source[0] = -phys_source[0]
            phys_source[1] = -phys_source[1]
            phys_target[0] = -phys_target[0]
            phys_target[1] = -phys_target[1]
        
        # 3. Apply the Affine Mapping
        ct_tensor = d[self.ref_key]
        affine = ct_tensor.affine.cpu().numpy()
        inv_affine = np.linalg.inv(affine)
        
        vox_source = nib.affines.apply_affine(inv_affine, phys_source)
        vox_target = nib.affines.apply_affine(inv_affine, phys_target)
        
        device = ct_tensor.device
        A = torch.tensor(vox_source, dtype=torch.float32, device=device) # Source
        B = torch.tensor(vox_target, dtype=torch.float32, device=device) # Target
        
        # 4. Generate 3D Grid
        _, X, Y, Z = ct_tensor.shape
        x_grid, y_grid, z_grid = torch.meshgrid(
            torch.arange(X, device=device), 
            torch.arange(Y, device=device), 
            torch.arange(Z, device=device), 
            indexing='ij'
        )
        P = torch.stack([x_grid, y_grid, z_grid], dim=-1).float() 
        
        # 5. Distance and Vector Math
        AB = B - A
        AP = P - A
        
        dot_AP_AB = torch.sum(AP * AB, dim=-1)
        dot_AB_AB = torch.sum(AB * AB, dim=-1)
        
        # t is the position along the line segment. 
        # t=0 is Source, t=1 is Target.
        t_raw = dot_AP_AB / (dot_AB_AB + 1e-8)
        
        # Clamp maximum at 1.0 to find the closest point strictly up to the target
        t_clamped = torch.clamp(t_raw, max=1.0)
        
        C = A + t_clamped.unsqueeze(-1) * AB
        dist_sq = torch.sum((P - C)**2, dim=-1)
        
        # 6. Apply Gaussian Spread
        gaussian_prior = torch.exp(-dist_sq / (2 * self.sigma**2))
        
        # 7. Bragg Peak Cutoff
        # Hard mask any voxel that is physically past the target point (t > 1.0)
        gaussian_prior[t_raw > 1.0] = 0.0
        
        d["geometric_prior"] = gaussian_prior.unsqueeze(0)
        
        return d
import torch
import torch.nn as nn
from omegaconf import OmegaConf

class BeamMaskedMAELoss(nn.Module):
    """
    Masked MAE loss evaluated per beam in the high-dose region.

    Args
    ----
    reduction : str
        'mean'  – average the per-beam losses across beams and batch (default)
        'sum'   – sum instead
        'none'  – return a tensor of shape (B, N) with one value per beam
    """

    def __init__(self,reduction: str = "mean"):
        super().__init__()
        self.threshold_pct = 0.10
        self.eps = 1e-8
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B, N, *spatial)
        target : (B, N, *spatial)

        Returns
        -------
        Scalar tensor (reduction='mean'/'sum') or (B, N) tensor (reduction='none').
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, "
                f"got {pred.shape} vs {target.shape}"
            )
        if pred.ndim < 3:
            raise ValueError(
                "Expected at least 3 dimensions (B, N, *spatial), "
                f"got {pred.ndim}"
            )

        B, N = pred.shape[:2]
        # Flatten spatial dims: (B, N, V)
        spatial_dims = pred.shape[2:]
        V = 1
        for d in spatial_dims:
            V *= d

        pred_flat   = pred.reshape(B, N, V)    # (B, N, V)
        target_flat = target.reshape(B, N, V)  # (B, N, V)

        # Per-beam max ground-truth dose: (B, N)
        beam_max = target_flat.amax(dim=-1)    # (B, N)

        # High-dose threshold per beam: (B, N, 1) for broadcasting
        threshold = (self.threshold_pct * beam_max).unsqueeze(-1)  # (B, N, 1)

        # Boolean mask: True where gt >= threshold  → shape (B, N, V)
        mask = target_flat >= threshold

        # Absolute error, zeroed outside the mask
        abs_err = (pred_flat - target_flat).abs()  # (B, N, V)
        masked_abs_err = abs_err * mask.float()

        # Sum of masked absolute errors and count of masked voxels per beam
        sum_err   = masked_abs_err.sum(dim=-1)   # (B, N)
        count     = mask.float().sum(dim=-1)     # (B, N)

        # Mean absolute error in the high-dose region, normalised by beam max
        # Safe division: if count == 0 or beam_max == 0, result is 0
        mean_abs_err  = sum_err / (count + self.eps)          # (B, N)
        normalised    = mean_abs_err / (beam_max + self.eps)  # (B, N)

        if self.reduction == "none":
            return normalised
        if self.reduction == "sum":
            return normalised.sum()
        return normalised.mean()  # default: 'mean'



class IDDCurveLoss(nn.Module):
    """
    IDD curve RMSE loss, normalised by the ground-truth IDD peak.
 
    Args
    ----
    beam_dim : int
        Which spatial dimension is the depth (beam) axis.
        Counted from the *spatial* part of the tensor, so:
          0 → first spatial dim (H for a (B, N, H, W) tensor)
          1 → second spatial dim (W) — default
        Negative indexing is also accepted:
          -1 → last spatial dim  (same as 1 for 2-D, same as 2 for 3-D)
    eps : float
        Denominator guard.
    reduction : str
        'mean' | 'sum' | 'none'.
        With 'none', returns shape (B, N) (or (B,) for single-channel input).
    """
 
    def __init__(self,beam_dim: int = -1,reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        self.beam_dim  = beam_dim
        self.eps       = 1e-8
        self.reduction = reduction
 
    # ------------------------------------------------------------------
    def _idd_curve(self, x: torch.Tensor, depth_dim: int) -> torch.Tensor:
        """
        Integrate x along every axis EXCEPT `depth_dim`.
        x shape : (B, N, *spatial)   [N may be 1]
        Returns : (B, N, D)  where D = x.shape[depth_dim]
        """
        n_spatial = x.ndim - 2          # strip B and N
        # Resolve negative depth_dim to positive (within spatial dims)
        pos_depth = depth_dim % n_spatial
 
        # Sum over all spatial dims except the depth dim
        # spatial axes in the full tensor are at positions 2, 3, ..., 1+n_spatial
        sum_axes = tuple(2 + i for i in range(n_spatial) if i != pos_depth)
        if sum_axes:
            idd = x.sum(dim=sum_axes)   # (B, N, D)
        else:
            idd = x                     # already (B, N, D) — 1-D spatial edge case
        return idd
 
    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, target : (B, N, *spatial) or (B, *spatial)
 
        Returns
        -------
        Scalar or (B, N) tensor depending on `reduction`.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, "
                f"got {pred.shape} vs {target.shape}"
            )
 
        # Normalise to (B, N, *spatial) — insert N=1 if needed
        squeezed = False
        if pred.ndim < 3:
            raise ValueError(
                f"Expected at least 3 dimensions (B, [N,] *spatial), got {pred.ndim}"
            )
        if pred.ndim == 3:
            # (B, H, W) → (B, 1, H, W)
            pred   = pred.unsqueeze(1)
            target = target.unsqueeze(1)
            squeezed = True
 
        # IDD curves: (B, N, D)
        idd_pred   = self._idd_curve(pred,   self.beam_dim)
        idd_target = self._idd_curve(target, self.beam_dim)
 
        # Per-beam IDD peak of ground truth: (B, N)
        idd_peak = idd_target.amax(dim=-1)
 
        # RMSD along depth dimension: (B, N)
        rmsd = ((idd_pred - idd_target) ** 2).mean(dim=-1).sqrt()
 
        # Normalise
        normalised = rmsd / (idd_peak + self.eps)  # (B, N)
 
        if squeezed:
            normalised = normalised.squeeze(1)      # (B,)
 
        if self.reduction == "none":
            return normalised
        if self.reduction == "sum":
            return normalised.sum()
        return normalised.mean()
 



import torch
import torch.nn as nn

class Stratified_plan_level_MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def stratified_plan_mae(self, pred_dose: torch.Tensor, gt_dose: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Level 2 Stratified Plan-Level MAE.
        Derives the prescription dose dynamically from the ground-truth volume.
        """
        rx_dose_proxy = gt_dose.max()
        abs_error = torch.abs(pred_dose - gt_dose)

        high_mask = gt_dose >= 0.80 * rx_dose_proxy
        mid_mask  = (gt_dose >= 0.30 * rx_dose_proxy) & (gt_dose < 0.80 * rx_dose_proxy)
        low_mask  = (gt_dose >= 0.10 * rx_dose_proxy) & (gt_dose < 0.30 * rx_dose_proxy)

        def get_stratum_mae(mask: torch.Tensor) -> torch.Tensor:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)
            
            return abs_error[mask].mean() / (rx_dose_proxy + self.eps)

        mae_high = get_stratum_mae(high_mask)
        mae_mid  = get_stratum_mae(mid_mask)
        mae_low  = get_stratum_mae(low_mask)

        valid_strata_count = (high_mask.sum() > 0).int() + \
                             (mid_mask.sum() > 0).int() + \
                             (low_mask.sum() > 0).int()

        if valid_strata_count == 0:
            return torch.tensor(0.0, device=pred_dose.device, dtype=pred_dose.dtype)

        plan_level_mae = (mae_high + mae_mid + mae_low) / valid_strata_count

        return plan_level_mae

    def forward(self, pred_dose: torch.Tensor, gt_dose: torch.Tensor) -> torch.Tensor:
        """
        Allows the class to be called directly like a standard PyTorch loss function.
        """
        return self.stratified_plan_mae(pred_dose, gt_dose)
    


def dim_ceil(val):
    return int(val) + (1 if val % 1 > 0 else 0)

class GammaLoss(nn.Module):
    def __init__(self, voxel_spacing=(3, 3, 4)):
        """
        Args:
            voxel_spacing (tuple): Physical size of a voxel in mm (dx, dy, dz).
        """
        super().__init__()
        # Fallback to default config if none provided
        self.voxel_spacing = voxel_spacing

    def local_gamma_3d(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Computes the 3D Local Gamma Pass Rate (1% / 1mm) on GPU.
        Supports 3D (D,H,W), 4D (C,D,H,W), or 5D (B,C,D,H,W) tensors.
        """
        device = pred.device
        dx, dy, dz = self.voxel_spacing
        
        dta_tol = 1.0        # 1.0 mm DTA criteria
        dose_tol_pct = 0.01  # 1% local dose criteria
        
        # Standardize input to 5D: (B, C, D, H, W)
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).unsqueeze(0)
            gt = gt.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 4:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            
        B, C, D, H, W = gt.shape
        
        # 1. Define evaluation mask (Voxels >= 10% of global maximum GT dose)
        gt_max = gt.max()
        eval_mask = gt >= 0.10 * gt_max
        
        total_eval_voxels = eval_mask.sum()
        if total_eval_voxels == 0:
            return torch.tensor(1.0, device=device) # Trivial pass

        # 2. Map tensor dimensions (D, H, W) to the correct spatial axes
        # Maintaining the openGate simulation rotation: D (slices) maps to y-axis.
        r_y = int(dim_ceil(dta_tol / dy))  # Radius for D dimension
        r_z = int(dim_ceil(dta_tol / dz))  # Radius for H dimension
        r_x = int(dim_ceil(dta_tol / dx))  # Radius for W dimension
        
        gamma_sq = torch.full_like(gt, float('inf'))
        
        # 3. Pad the PREDICTION directly
        pad_w = (r_x, r_x, r_z, r_z, r_y, r_y) # Padding for (W, H, D)
        pred_padded = torch.nn.functional.pad(pred, pad_w, mode='replicate')
        
        # Pre-compute the ground truth dose tolerance mapping
        dose_tol_map = dose_tol_pct * gt
        dose_tol_sq = (dose_tol_map + 1e-8)**2
        
        # 4. Search over the localized 3D neighborhood
        for sy in range(-r_y, r_y + 1):
            for sz in range(-r_z, r_z + 1):
                for sx in range(-r_x, r_x + 1):
                    
                    # Physical distance based on the axis mapping
                    dist_sq = (sy * dy)**2 + (sz * dz)**2 + (sx * dx)**2
                    if dist_sq > dta_tol**2: 
                        continue 
                    
                    # Normalize the distance by the DTA criteria squared
                    spatial_term = dist_sq / (dta_tol**2)
                    
                    # Shift the evaluated volume against the static reference, preserving B and C dimensions
                    pred_shifted = pred_padded[
                        :, :,
                        r_y + sy : r_y + sy + D,
                        r_z + sz : r_z + sz + H,
                        r_x + sx : r_x + sx + W
                    ]
                    
                    # Compute dose discrepancy using the fixed GT tolerance
                    dose_diff_sq = ((pred_shifted - gt)**2) / dose_tol_sq
                    
                    # Gamma index addition
                    current_gamma_sq = spatial_term + dose_diff_sq
                    gamma_sq = torch.minimum(gamma_sq, current_gamma_sq)

        # 5. Extract final gamma metrics inside the valid evaluation mask
        gamma = torch.sqrt(gamma_sq[eval_mask])
        
        passed_voxels = (gamma <= 1.0).sum()
        pass_rate = passed_voxels.float() / total_eval_voxels.float()
        
        return 1.0-pass_rate
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return self.local_gamma_3d(pred, gt)


class TotalVariationLoss3D(nn.Module):
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        # x shape: [Batch, Channel, X, Y, Z]
        # Calculate absolute differences between adjacent voxels in all 3 spatial dimensions
        tv_x = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
        tv_y = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
        tv_z = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
        
        return self.weight * (tv_x + tv_y + tv_z)
    

def gradient_difference_loss_3d(pred, target):
    # Calculate gradients (edges) in X, Y, and Z
    pred_dx = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
    pred_dy = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    pred_dz = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    
    target_dx = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
    target_dy = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
    target_dz = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
    
    # Penalize the network if its edges are less sharp than the ground truth edges
    loss = torch.mean(torch.abs(pred_dx - target_dx)) + \
           torch.mean(torch.abs(pred_dy - target_dy)) + \
           torch.mean(torch.abs(pred_dz - target_dz))
           
    return loss


class LossEvaluator(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _get_d_x(self, dose_volume: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        """Helper to calculate the Dose to X% of the volume (quantile)."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=dose_volume.device)
        
        # Extract doses inside the mask and ensure they are float32 for torch.quantile
        doses_in_structure = dose_volume[mask].float()
        return torch.quantile(doses_in_structure, q)

    def _get_v_x(self, dose_volume: torch.Tensor, mask: torch.Tensor, threshold_dose: float) -> torch.Tensor:
        """Helper to calculate Volume receiving >= threshold_dose (fractional)."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=dose_volume.device)
        
        doses_in_structure = dose_volume[mask]
        # Calculate the fraction of voxels meeting the threshold
        return (doses_in_structure >= threshold_dose).float().mean()

    def _get_d_mean(self, dose_volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Helper to calculate Mean Dose."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=dose_volume.device)
        
        return dose_volume[mask].mean()

    def _absolute_relative_diff(self, pred_val: torch.Tensor, gt_val: torch.Tensor) -> torch.Tensor:
        """Computes |Pred - GT| / GT, safeguarded with epsilon."""
        return torch.abs(pred_val - gt_val) / (gt_val + self.eps)

    def dvh_clinical_score(self, 
                           pred_dose: torch.Tensor, 
                           gt_dose: torch.Tensor, 
                           ptv_mask: torch.Tensor, 
                           oar_masks, rx_dose  = None) -> torch.Tensor:
        """
        Computes the standardized Level 2.3 DVH-Based Clinical Score.
        
        Args:
            pred_dose (torch.Tensor): Predicted dose volume.
            gt_dose (torch.Tensor): Ground-truth dose volume.
            ptv_mask (torch.Tensor): Boolean mask for the Target (PTV).
            oar_masks (List[torch.Tensor]): List of boolean masks for the 3 closest OARs.
            rx_dose (float, optional): Prescribed dose. If None, uses max of gt_dose.
            
        Returns:
            torch.Tensor: The final DVH score (lower is better, 0.0 is perfect).
        """
        device = pred_dose.device
        if rx_dose is None:
            rx_dose = gt_dose.max().item()

        # ---------------------------------------------------------
        # 1. Target (PTV) Metrics
        # ---------------------------------------------------------
        # D98% (2nd percentile)
        ptv_d98_gt = self._get_d_x(gt_dose, ptv_mask, q=0.02)
        ptv_d98_pred = self._get_d_x(pred_dose, ptv_mask, q=0.02)
        ard_ptv_d98 = self._absolute_relative_diff(ptv_d98_pred, ptv_d98_gt)

        # V95% (Volume >= 95% of Rx)
        v95_threshold = 0.95 * rx_dose
        ptv_v95_gt = self._get_v_x(gt_dose, ptv_mask, v95_threshold)
        ptv_v95_pred = self._get_v_x(pred_dose, ptv_mask, v95_threshold)
        ard_ptv_v95 = self._absolute_relative_diff(ptv_v95_pred, ptv_v95_gt)

        # Average Target Score
        target_score = (ard_ptv_d98 + ard_ptv_v95) / 2.0

        # ---------------------------------------------------------
        # 2. OAR Metrics (Iterate through the 3 OARs)
        # ---------------------------------------------------------
        oar_ards = []
        for oar_mask in oar_masks:
            if oar_mask.sum() == 0:
                continue # Skip if an OAR mask is entirely empty in this patch/volume

            # D2% (98th percentile)
            oar_d2_gt = self._get_d_x(gt_dose, oar_mask, q=0.98)
            oar_d2_pred = self._get_d_x(pred_dose, oar_mask, q=0.98)
            ard_oar_d2 = self._absolute_relative_diff(oar_d2_pred, oar_d2_gt)

            # D_mean
            oar_dmean_gt = self._get_d_mean(gt_dose, oar_mask)
            oar_dmean_pred = self._get_d_mean(pred_dose, oar_mask)
            ard_oar_dmean = self._absolute_relative_diff(oar_dmean_pred, oar_dmean_gt)

            oar_ards.extend([ard_oar_d2, ard_oar_dmean])

        # Average OAR Score
        if len(oar_ards) > 0:
            oar_score = torch.stack(oar_ards).mean()
        else:
            oar_score = torch.tensor(0.0, device=device)

        # ---------------------------------------------------------
        # 3. Final Combined Score (Equal Weighting)
        # ---------------------------------------------------------
        # The prompt specifies equal contribution from target metrics and OAR metrics.
        # Target weight = 0.5, OAR weight = 0.5
        dvh_score = (target_score * 0.5) + (oar_score * 0.5)

        return dvh_score
    
import torch.nn.functional as F

class SimpleMaskedMAE(nn.Module):
    def __init__(self, threshold_pct=0.1):
        super().__init__()
        self.threshold = threshold_pct

    def forward(self, pred, target):
        # 1. Find the max dose for each volume in the batch
        # Keep dims so it broadcasts correctly: (B, 1, 1, 1, 1)
        batch_max = target.amax(dim=(2, 3, 4), keepdim=True)
        
        # 2. Create a boolean mask of the high-dose region
        mask = target >= (batch_max * self.threshold)
        
        # 3. Failsafe: If mask is somehow empty, just do a global MAE
        if mask.sum() == 0:
            return F.l1_loss(pred, target)
            
        # 4. Extract only the masked voxels (flattens into a 1D list automatically)
        # and compute the native PyTorch mean absolute error.
        return F.l1_loss(pred[mask], target[mask])


class SimpleIDDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 1. Collapse the lateral spatial dimensions (X=2, Y=3) using MEAN
        # This reduces (B, 1, 64, 64, 32) -> (B, 1, 32)
        pred_curve = pred.mean(dim=(2, 3))
        target_curve = target.mean(dim=(2, 3))
        
        # 2. Compute the mean absolute error between the two 1D curves
        return F.l1_loss(pred_curve, target_curve)



import torch


def compute_angle_agnostic_idd(dose_volume, ray_source, ray_target, spacing=(4.0, 4.0, 3.0), bin_size_mm=4.0):
    """
    dose_volume: (B, 1, X, Y, Z) - The predicted or ground truth dose
    ray_source: (B, 3) - Physical coordinates of the beam source
    ray_target: (B, 3) - Physical coordinates of the beam target
    spacing: Tuple of the physical voxel dimensions
    bin_size_mm: The resolution of the 1D output curve
    """
    B, _, X, Y, Z = dose_volume.shape
    device = dose_volume.device
    
    # 1. Create the 3D grid of physical coordinates (matching your Resize dimensions)
    # Shape of each: (X, Y, Z)
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(X, device=device) * spacing[0],
        torch.arange(Y, device=device) * spacing[1],
        torch.arange(Z, device=device) * spacing[2],
        indexing='ij'
    )
    # Stack into a single coordinate tensor: (3, X, Y, Z)
    coords = torch.stack([grid_x, grid_y, grid_z], dim=0)
    # Flatten spatial dimensions for easier batch math: (B, 3, N_voxels)
    coords = coords.view(1, 3, -1).expand(B, -1, -1) 
    
    # Flatten the dose volume: (B, N_voxels)
    dose_flat = dose_volume.view(B, -1)
    
    # 2. Calculate the normalized beam vector
    beam_vec = ray_target - ray_source # (B, 3)
    beam_len = torch.norm(beam_vec, dim=-1, keepdim=True)
    beam_dir = beam_vec / (beam_len + 1e-6) # (B, 3)
    
    # 3. Project voxel coordinates onto the beam line to get depth
    # Reshape for broadcasting: source (B, 3, 1), dir (B, 3, 1)
    source_b = ray_source.unsqueeze(-1)
    dir_b = beam_dir.unsqueeze(-1)
    
    # Depth = dot_product(coords - source, beam_dir)
    # Resulting shape: (B, N_voxels)
    depths = torch.sum((coords - source_b) * dir_b, dim=1)
    
    # 4. Discretize depths into integer bins (e.g., every 2mm)
    # We clamp at 0 to ignore voxels located "behind" the source
    depth_bins = (depths / bin_size_mm).clamp(min=0).long()
    
    # Define the maximum length of the 1D curve (e.g., 400mm / 2mm bins = 200 bins)
    num_bins = 200 
    depth_bins = depth_bins.clamp(max=num_bins - 1)
    
    # 5. Integrate (Sum) the dose laterally using scatter_add
    # This collapses the 3D volume into a 1D curve per batch item
    idd_curves = torch.zeros((B, num_bins), dtype=torch.float32, device=device)
    idd_curves.scatter_add_(dim=1, index=depth_bins, src=dose_flat)
    
    return idd_curves
    

class BraggPeakPositionLoss(nn.Module):
    def __init__(self, temperature=10.0):
        super().__init__()
        # A higher temperature makes the peak isolation sharper.
        # 10.0 is usually a sweet spot for suppressing the dose plateau.
        self.temperature = temperature 

    def forward(self, pred, target):
        """
        pred, target shape: (Batch, 1, X, Y, Z)
        """
        B, _, X, Y, Z = pred.shape
        device = pred.device
        
        # 1. Scale the volumes between 0 and 1
        # We add 1e-6 to prevent division by zero in empty volumes
        pred_max = pred.amax(dim=(2, 3, 4), keepdim=True) + 1e-6
        target_max = target.amax(dim=(2, 3, 4), keepdim=True) + 1e-6
        
        pred_scaled = F.relu(pred) / pred_max
        target_scaled = F.relu(target) / target_max
        
        # 2. Isolate the Bragg Peak using the temperature power
        # The plateau vanishes, leaving only the high-dose peak
        pred_weight = torch.pow(pred_scaled, self.temperature)
        target_weight = torch.pow(target_scaled, self.temperature)
        
        # 3. Normalize into a spatial probability distribution (sums to 1.0)
        pred_prob = pred_weight / pred_weight.sum(dim=(2, 3, 4), keepdim=True)
        target_prob = target_weight / target_weight.sum(dim=(2, 3, 4), keepdim=True)
        
        # 4. Generate Coordinate Grids (Normalized from -1 to 1 for numerical stability)
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.linspace(-1, 1, X, device=device),
            torch.linspace(-1, 1, Y, device=device),
            torch.linspace(-1, 1, Z, device=device),
            indexing='ij'
        )
        # Reshape to (1, 3, X, Y, Z) to broadcast across the batch
        grid = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
        
        # 5. Calculate Center of Mass (The Expected 3D Coordinate)
        # Multiply the probability map by the grid and sum the spatial dimensions
        pred_com = (pred_prob.unsqueeze(1) * grid).sum(dim=(3, 4, 5))    # Shape: (B, 3)
        target_com = (target_prob.unsqueeze(1) * grid).sum(dim=(3, 4, 5)) # Shape: (B, 3)
        
        # 6. Calculate the L2 Distance (MSE) between the predicted and true peak positions
        position_loss = F.mse_loss(pred_com, target_com)
        
        return position_loss

class Level1LossFunction(nn.Module):
    def __init__(self,masked_factor=1.0, iid_curve_weight=0.001, allMAE_weight=1.0, use_high_dose_mask=False, high_dose_threshold=0.8, high_dose_weight=1.0,bragg_peak_weight=50.0):
        super().__init__()
        self.beam_masked_mae_loss = SimpleMaskedMAE()
        self.bragg_peak_loss = BraggPeakPositionLoss()
        self.allMAE = torch.nn.L1Loss()
        if use_high_dose_mask:
            self.beam_masked_mae_loss = SimpleMaskedMAE(threshold_pct=high_dose_threshold)
            self.high_dose_weight = high_dose_weight
        self.masked_factor = masked_factor
        self.iid_curve_weight = iid_curve_weight
        self.allMAE_weight = allMAE_weight
        self.use_high_dose_mask = use_high_dose_mask
        self.bragg_peak_weight = bragg_peak_weight
    
    def __call__(self, pred_dose, gt_dose,ray_source=None, ray_target=None, use_bragg_peak_loss=True,use_idd_loss=True):
        beam_masked_mae = self.beam_masked_mae_loss(pred_dose, gt_dose)
        #idd_curve_loss_value = #self.IID_curve_loss(pred_dose, gt_dose)
        allMAE = self.allMAE(pred_dose, gt_dose)
        eff_masked = beam_masked_mae * self.masked_factor
        eff_all = allMAE * self.allMAE_weight
        total_loss = eff_masked  + eff_all

        if ray_source is not None and ray_target is not None:
            pred_idd = compute_angle_agnostic_idd(pred_dose, ray_source, ray_target)
            target_idd = compute_angle_agnostic_idd(gt_dose, ray_source, ray_target)
            idd_mae = F.mse_loss(pred_idd, target_idd)
            eff_idd = idd_mae * self.iid_curve_weight
       
        if self.use_high_dose_mask:
            high_beam_masked_mae = self.beam_masked_mae_loss(pred_dose, gt_dose)
            eff_high_masked = high_beam_masked_mae * self.high_dose_weight
            total_loss = total_loss + eff_high_masked
        
        lossDict = {
            "masked_mae": beam_masked_mae,
            "idd_curve_loss_value": idd_mae,
            "allMAE": allMAE,
            "effective_beam_masked_mae": eff_masked,
            "effective_idd_curve_loss": eff_idd,
            "effective_allMAE": eff_all,
            "total_loss": total_loss
        }

        if use_idd_loss:
            total_loss = total_loss+ eff_idd
        if use_bragg_peak_loss:
            bragg_peak_loss = self.bragg_peak_loss(pred_dose, gt_dose)
            eff_bragg = bragg_peak_loss * self.bragg_peak_weight  # You can adjust the weight for Bragg Peak Loss if needed
            lossDict["bragg_peak_loss"] = bragg_peak_loss
            lossDict["effective_bragg_peak_loss"] = eff_bragg
            total_loss = total_loss + eff_bragg
            lossDict["total_loss"] = total_loss

        if self.use_high_dose_mask:
            lossDict["high_beam_masked_mae"] = high_beam_masked_mae
            lossDict["effective_high_beam_masked_mae"] = eff_high_masked
            
        return lossDict
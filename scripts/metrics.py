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

class LossEvaluator(nn.Module):
    def __init__(self, voxel_spacing = None):
        """
        Args:
            voxel_spacing (tuple): Physical size of a voxel in mm (dx, dy, dz).
                                   Crucial for calculating Distance-to-Agreement.
        """
        super().__init__()
        cfg = OmegaConf.load("configs/default_config.yaml")
        self.voxel_spacing = cfg['data']['voxel_spacing'] if voxel_spacing is None else voxel_spacing

    def local_gamma_3d(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Computes the 3D Local Gamma Pass Rate (1% / 1mm) on GPU.
        
        Args:
            pred (torch.Tensor): Predicted dose volume, shape (D, H, W)
            gt (torch.Tensor): Ground truth dose volume, shape (D, H, W)
            
        Returns:
            torch.Tensor: Scalar tensor containing the pass rate (0.0 to 1.0)
        """
        device = pred.device
        dx, dy, dz = self.voxel_spacing
        
        # 1. Define evaluation mask (Voxels >= 10% of global maximum GT dose)
        gt_max = gt.max()
        eval_mask = gt >= 0.10 * gt_max
        
        total_eval_voxels = eval_mask.sum()
        if total_eval_voxels == 0:
            return torch.tensor(1.0, device=device) # Trivial pass if no high dose regions

        # 2. Determine search window radius in voxels based on 1.0 mm limit
        # A voxel distance further than 1mm automatically makes the spatial term > 1,
        # meaning gamma cannot be <= 1. Thus, we only search within a 1mm radius.
        r_z = int(dim_ceil(1.0 / dz))
        r_y = int(dim_ceil(1.0 / dy))
        r_x = int(dim_ceil(1.0 / dx))
        
        # Initialize gamma matrix with infinity
        gamma_sq = torch.full_like(gt, float('inf'))
        
        # Pad tensors to safely handle edge shifts
        pad_w = (r_x, r_x, r_y, r_y, r_z, r_z)
        gt_padded = torch.nn.functional.pad(gt, pad_w, mode='edge')
        
        D, H, W = gt.shape

        # 3. Search over the localized 3D neighborhood
        for sz in range(-r_z, r_z + 1):
            for sy in range(-r_y, r_y + 1):
                for sx in range(-r_x, r_x + 1):
                    # Calculate physical Euclidean distance
                    dist_sq = (sz * dz)**2 + (sy * dy)**2 + (sx * dx)**2
                    if dist_sq > 1.0: 
                        continue # Skip neighbor if distance strictly exceeds 1 mm
                    
                    # Crop the shifted reference region matching the original dimensions
                    gt_shifted = gt_padded[
                        r_z + sz : r_z + sz + D,
                        r_y + sy : r_y + sy + H,
                        r_x + sx : r_x + sx + W
                    ]
                    
                    # Local dose difference tolerance: 1% of the LOCAL reference dose
                    dose_tol = 0.01 * gt_shifted
                    
                    # Compute dose discrepancy term safely avoiding division by zero
                    dose_diff_sq = ((pred - gt_shifted) / (dose_tol + 1e-8))**2
                    
                    # Combined Gamma space value for this specific neighbor displacement
                    current_gamma_sq = dist_sq + dose_diff_sq
                    
                    # Retain the minimum found across all checked neighbors
                    gamma_sq = torch.minimum(gamma_sq, current_gamma_sq)

        # 4. Extract final gamma metrics inside the valid evaluation mask
        gamma = torch.sqrt(gamma_sq[eval_mask])
        
        # Pass rate is defined as the percentage of voxels where gamma <= 1
        passed_voxels = (gamma <= 1.0).sum()
        pass_rate = passed_voxels.float() / total_eval_voxels.float()
        
        return pass_rate

def test_beam_masked_mae():
    torch.manual_seed(0)
    B, N, D, H, W = 2, 3, 16, 64, 64

    target = torch.rand(B, N, D, H, W)
    pred   = target + 0.05 * torch.randn(B, N, D, H, W)

    loss_fn = BeamMaskedMAELoss()
    loss    = loss_fn(pred, target)
    print(f"Loss (mean, 3D): {loss.item():.6f}")

    # reduction='none' → per-beam tensor
    per_beam = BeamMaskedMAELoss(reduction="none")(pred, target)
    print(f"Per-beam shape: {per_beam.shape}")   # (2, 3)
    print(f"Per-beam values:\n{per_beam}")

    # 2-D spatial (e.g. slice-wise)
    target_2d = torch.rand(B, N, H, W)
    pred_2d   = target_2d + 0.05 * torch.randn(B, N, H, W)
    loss_2d   = loss_fn(pred_2d, target_2d)
    print(f"Loss (mean, 2D): {loss_2d.item():.6f}")

    # Zero-dose beam edge case
    target_zero          = torch.rand(B, N, D, H, W)
    target_zero[:, 0, :] = 0.0   # first beam has zero gt dose
    pred_zero            = torch.rand(B, N, D, H, W)
    loss_zero            = loss_fn(pred_zero, target_zero)
    print(f"Loss with one zero-dose beam: {loss_zero.item():.6f}")


def test_iid():
    import torch
 
    torch.manual_seed(42)
    B, N, H, W = 2, 3, 64, 64
 
    # ---- basic forward pass ----
    target = torch.rand(B, N, H, W)
    pred   = target + 0.05 * torch.randn(B, N, H, W)
 
    loss_fn = IDDCurveLoss(beam_dim=-1)           # depth = W axis
    loss    = loss_fn(pred, target)
    print(f"Loss (mean, beam_dim=-1): {loss.item():.6f}")
 
    loss_fn0 = IDDCurveLoss(beam_dim=0)           # depth = H axis
    loss0    = loss_fn0(pred, target)
    print(f"Loss (mean, beam_dim= 0): {loss0.item():.6f}")
 
    # ---- perfect prediction → loss should be ~0 ----
    loss_perfect = loss_fn(target, target)
    print(f"Loss (perfect pred):      {loss_perfect.item():.6f}")
 
    # ---- reduction='none' → per-beam ----
    per_beam = IDDCurveLoss(beam_dim=-1, reduction="none")(pred, target)
    print(f"Per-beam shape: {per_beam.shape}")    # (B, N) = (2, 3)
    print(f"Per-beam values:\n{per_beam}")
 
    # ---- single-channel total dose (B, H, W) ----
    target_2d = torch.rand(B, H, W)
    pred_2d   = target_2d + 0.05 * torch.randn(B, H, W)
    loss_2d   = IDDCurveLoss(beam_dim=-1)(pred_2d, target_2d)
    print(f"Loss (single-channel 2D): {loss_2d.item():.6f}")
 
    # ---- zero ground truth edge case ----
    target_z  = torch.zeros(B, N, H, W)
    pred_z    = torch.rand(B, N, H, W)
    loss_z    = loss_fn(pred_z, target_z)
    print(f"Loss (zero gt):           {loss_z.item():.6f}")   # should be large but finite

if __name__ == "__main__":
    test_iid()
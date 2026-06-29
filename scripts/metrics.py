import torch
import torch.nn as nn

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
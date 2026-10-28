#!/usr/bin/env python3
"""Visualize predicted and ground-truth dose volumes (batch size 1).

Usage examples:
  python visualize_dose.py --pred path/to/pred.nii.gz --gt path/to/gt.nii.gz
  python visualize_dose.py --pred pred.npy --gt gt.npy --napari

The script attempts to use MONAI's `plot_2d_or_3d_image` if available,
falls back to matplotlib orthogonal slices, and can open an interactive
Napari viewer when requested.
"""
import argparse
import os
import numpy as np


def load_volume(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nii", ".gz") or path.endswith(".nii.gz"):
        try:
            import nibabel as nib

            arr = nib.load(path).get_fdata()
        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI file {path}: {e}")
    elif ext == ".npy":
        arr = np.load(path)
    elif ext in (".pt", ".pth"):
        try:
            import torch

            obj = torch.load(path)
            arr = obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.array(obj)
        except Exception as e:
            raise RuntimeError(f"Failed to load torch file {path}: {e}")
    else:
        # generic try
        try:
            arr = np.load(path)
        except Exception:
            raise RuntimeError(f"Unsupported file type for {path}")

    # Normalize shape to a single 3D array (D,H,W)
    if arr.ndim == 5:  # (B,C,D,H,W) or (B,D,H,W,C) unlikely
        arr = arr[0]
    if arr.ndim == 4:
        # Could be (C,D,H,W) or (B,D,H,W) or (D,H,W,1)
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1 and arr.ndim == 4:
            arr = arr[..., 0]
        elif arr.shape[0] > 1 and arr.shape[0] < 5:
            # treat as (C,D,H,W) -> pick channel 0
            arr = arr[0]
        else:
            # treat as (B,D,H,W)
            arr = arr[0]
    if arr.ndim != 3:
        raise RuntimeError(f"Loaded array for {path} is not 3D after normalization (shape={arr.shape})")
    return arr.astype(float)


def compute_metrics(pred, gt):
    diff = pred - gt
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mre = float(np.mean(np.abs(diff) / (np.abs(gt) + 1e-8)))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MeanRelError": mre}


def show_matplotlib_slices(pred, gt, cmap="viridis"):
    import matplotlib.pyplot as plt

    # show middle axial, coronal, sagittal
    dz, dy, dx = pred.shape
    slices = [
        (pred[dz // 2], gt[dz // 2], f"Axial (z={dz//2})"),
        (pred[:, dy // 2, :], gt[:, dy // 2, :], f"Coronal (y={dy//2})"),
        (pred[:, :, dx // 2], gt[:, :, dx // 2], f"Sagittal (x={dx//2})"),
    ]

    fig, axs = plt.subplots(3, 2, figsize=(10, 14))
    for i, (p_s, g_s, title) in enumerate(slices):
        axs[i, 0].imshow(np.rot90(p_s), cmap=cmap)
        axs[i, 0].set_title(f"Pred - {title}")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(np.rot90(g_s), cmap=cmap)
        axs[i, 1].set_title(f"GT - {title}")
        axs[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


def try_monai_plot(pred, gt):
    try:
        from monai.visualize import plot_2d_or_3d_image

        # MONAI expects either a single array or a list/tuple of arrays
        imgs = [pred, gt]
        titles = ["pred", "gt"]
        plot_2d_or_3d_image(imgs, None, None, titles=titles)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Visualize predicted and ground-truth dose volumes (batch size 1)")
    parser.add_argument("--pred", required=True, help="Path to predicted dose (npy, nii.gz, pt)")
    parser.add_argument("--gt", required=True, help="Path to ground-truth dose (npy, nii.gz, pt)")
    parser.add_argument("--napari", action="store_true", help="Open Napari for interactive 3D visualization if available")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap for slices")
    args = parser.parse_args()

    pred = load_volume(args.pred)
    gt = load_volume(args.gt)

    # Ensure same shape
    if pred.shape != gt.shape:
        raise RuntimeError(f"Shape mismatch pred {pred.shape} vs gt {gt.shape}")

    metrics = compute_metrics(pred, gt)
    print("Computed metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g}")

    # Try MONAI plotting first
    used_monai = False
    try:
        used_monai = try_monai_plot(pred, gt)
    except Exception:
        used_monai = False

    if not used_monai:
        print("MONAI plotting unavailable or failed — showing matplotlib slices.")
        show_matplotlib_slices(pred, gt, cmap=args.cmap)

    if args.napari:
        try:
            import napari

            v = napari.Viewer()
            # add volumes side-by-side with transparency
            v.add_image(pred, name="pred", colormap="viridis")
            v.add_image(gt, name="gt", colormap="magma", opacity=0.6)
            napari.run()
        except Exception as e:
            print(f"Failed to open napari viewer: {e}")


if __name__ == "__main__":
    main()

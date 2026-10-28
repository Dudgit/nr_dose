import nibabel as nib
import numpy as np
import torch


def _to_numpy_xyz(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)[:3]


def place_prediction_slab_in_full_volume(
    pred_slab,
    reference_img,
    ray_source,
    ray_target,
    center_mode="source",
):
    if hasattr(pred_slab, "as_tensor"):
        pred = pred_slab.as_tensor().clone()
    elif torch.is_tensor(pred_slab):
        pred = pred_slab.clone()
    else:
        pred = torch.as_tensor(pred_slab)

    if pred.ndim == 3:
        pred = pred.unsqueeze(0)

    if pred.ndim != 4:
        raise ValueError(
            "pred_slab shape-je (C, X, Y, Z) vagy (X, Y, Z) legyen, "
            f"kapott shape: {tuple(pred.shape)}"
        )

    x_full, y_full, z_full = map(int, reference_img.shape[-3:])
    c_pred, x_pred, y_pred, z_pred = map(int, pred.shape)

    if (x_pred, y_pred) != (x_full, y_full):
        raise ValueError(
            "A predikció XY-mérete nem egyezik a referencia XY-méretével: "
            f"pred={(x_pred, y_pred)}, reference={(x_full, y_full)}"
        )

    source = _to_numpy_xyz(ray_source)
    target = _to_numpy_xyz(ray_target)

    if center_mode == "source":
        physical_center = source
    elif center_mode == "target":
        physical_center = target
    elif center_mode == "midpoint":
        physical_center = 0.5 * (source + target)
    else:
        raise ValueError(
            "center_mode csak 'source', 'target' vagy 'midpoint' lehet."
        )

    affine = reference_img.affine
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    else:
        affine = np.asarray(affine)

    voxel_center = nib.affines.apply_affine(
        np.linalg.inv(affine),
        physical_center,
    )
    z_index = int(np.round(voxel_center[2]))

    z_start = z_index - z_pred // 2
    z_end = z_start + z_pred

    dst_z_start = max(0, z_start)
    dst_z_end = min(z_full, z_end)
    src_z_start = max(0, -z_start)
    copy_length = max(0, dst_z_end - dst_z_start)
    src_z_end = src_z_start + copy_length

    full_prediction = torch.zeros(
        (c_pred, x_full, y_full, z_full),
        dtype=pred.dtype,
        device=pred.device,
    )

    if copy_length:
        full_prediction[..., dst_z_start:dst_z_end] = (
            pred[..., src_z_start:src_z_end]
        )

    return full_prediction


class DoseCallWrapper:
    def __init__(
        self,
        model,
        transforms,
        inverse,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.transforms = transforms
        self.inverse = inverse

    def __call__(self, train_dicts):
        samples = [
            self.transforms(dict(train_dict))
            for train_dict in train_dicts
        ]

        x = torch.stack([
            torch.cat(
                [
                    sample["ct"],
                    sample["geometric_prior"],
                    sample["field"],
                ],
                dim=0,
            )
            for sample in samples
        ]).to(self.device)

        condition = torch.stack([
            torch.as_tensor(
                sample["condition"],
                dtype=torch.float32,
            )
            for sample in samples
        ]).to(self.device)

        if condition.ndim == 1:
            condition = condition.unsqueeze(1)

        with torch.inference_mode():
            pred = self.model(x, condition)

        true_preds = []

        for sample, sample_pred in zip(samples, pred):
            sample["pred"] = sample_pred.detach().cpu()
            true_preds.append(
                self.inverse(sample)["pred"]
            )

        return true_preds
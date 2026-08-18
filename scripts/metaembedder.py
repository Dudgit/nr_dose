import numpy as np
import torch
import nibabel as nib
from monai.transforms import MapTransform


class EnergyInjector(MapTransform):
    """
    THIS IS JUST A SAMPLE TEMPLATE!!!!
    """
    def __init__(self, keys, energy_key="energy", ref_key="ct"):
        super().__init__(keys)
        self.energy_key = energy_key
        self.ref_key = ref_key

    def __call__(self, data):
        d = dict(data)
        energy_value = d[self.energy_key]
        ct_tensor = d[self.ref_key]
        
        # Create a tensor filled with the energy value, matching the CT tensor's shape
        energy_tensor = torch.full_like(ct_tensor, fill_value=energy_value)
        
        d["energy_map"] = energy_tensor
        
        return d


import torch
import torch.nn as nn
import math

class FourierFeatureEmbedder(nn.Module):
    def __init__(self, in_features=1, num_freqs=8, include_input=True):
        """
        Args:
            in_features: The length of the raw input vector (1 if just energy).
            num_freqs: The number of frequency bands (L). Higher means it can resolve 
                       finer numerical differences, but costs more channels.
            include_input: Whether to append the raw unencoded scalar to the output.
        """
        super().__init__()
        self.in_features = in_features
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Calculate the final output dimension for the MLP mathematically
        self.out_dim = 0
        if include_input:
            self.out_dim += in_features
        # For each frequency band, we generate both a sine and cosine projection
        self.out_dim += in_features * num_freqs * 2 
        
        # Generate the frequency bands: [2^0, 2^1, 2^2, ..., 2^(L-1)]
        # We use register_buffer so PyTorch handles DDP device placement automatically
        freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        """
        x shape: (Batch, in_features)
        Returns: (Batch, out_dim)
        """
        embeds = []
        if self.include_input:
            embeds.append(x)
            
        for freq in self.freq_bands:
            # Multiply the input by the frequency scalar and pi
            val = x * freq * math.pi
            embeds.append(torch.sin(val))
            embeds.append(torch.cos(val))
            
        # Concatenate everything along the feature dimension
        return torch.cat(embeds, dim=-1)


import torch
from monai.transforms import MapTransform

class InjectEnergyDepositionFieldd(MapTransform):
    def __init__(
        self,
        keys,
        ref_key="ct",
        source_key="ray_source",
        target_key="ray_target",
        energy_key="condition",
        spacing=(4.0, 4.0, 3.0),
        sigma_base=2.0,
        alpha=0.004,
        plateau_ratio=0.70,
        sigma_entrance=16.0,
        sigma_distal=8.0,
        allow_missing_keys=False
    ):
        super().__init__(keys, allow_missing_keys)
        self.ref_key = ref_key
        self.source_key = source_key
        self.target_key = target_key
        self.energy_key = energy_key
        self.spacing = spacing
        self.sigma_base = sigma_base
        self.alpha = alpha
        self.plateau_ratio = plateau_ratio
        self.sigma_entrance = sigma_entrance
        self.sigma_distal = sigma_distal

    def __call__(self, data):
        d = dict(data)
        ref_img = d[self.ref_key]
        
        if not isinstance(ref_img, torch.Tensor):
            ref_img = torch.as_tensor(ref_img)
        device = ref_img.device
        dtype = ref_img.dtype
        
        spatial_shape = ref_img.shape[-3:]
        X, Y, Z = spatial_shape
        
        # 1. Extract physical vectors and scalar energy
        source_global = torch.as_tensor(d[self.source_key], dtype=dtype, device=device)
        target_global = torch.as_tensor(d[self.target_key], dtype=dtype, device=device)
        energy = torch.as_tensor(d[self.energy_key], dtype=dtype, device=device)
        
        # 2. Calculate normalized unit beam direction vector from global points
        beam_vec_global = target_global - source_global
        d_target = torch.norm(beam_vec_global) + 1e-6
        beam_dir = beam_vec_global / d_target  # Direction vector (dx, dy, dz)
        
        # 3. Define LOCAL slab physical coordinates
        # Grid origin starts at 0 for the local cropped volume
        grid_x = torch.arange(X, device=device, dtype=dtype) * self.spacing[0]
        grid_y = torch.arange(Y, device=device, dtype=dtype) * self.spacing[1]
        grid_z = torch.arange(Z, device=device, dtype=dtype) * self.spacing[2]
        coords = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=0)
        
        # 4. Center local target at the exact middle of the cropped volume (16th slab)
        target_local = torch.tensor([
            (X * self.spacing[0]) / 2.0,
            (Y * self.spacing[1]) / 2.0,
            (Z * self.spacing[2]) / 2.0
        ], dtype=dtype, device=device)
        
        # Local source is reconstructed backwards along the beam direction
        source_local = target_local - (d_target * beam_dir)
        
        # 5. Project LOCAL coordinates onto local beam line
        v = coords - source_local.view(3, 1, 1, 1)
        depths = torch.sum(v * beam_dir.view(3, 1, 1, 1), dim=0)
        
        # 6. Lateral distance squared (r^2) from ray axis
        v_lateral = v - (depths.unsqueeze(0) * beam_dir.view(3, 1, 1, 1))
        r_sq = torch.sum(v_lateral ** 2, dim=0)
        
        # 7. Physical profiles
        depths_pos = torch.clamp(depths, min=0.0)
        sigma_d = self.sigma_base + (self.alpha * depths_pos)
        lateral_profile = torch.exp(-r_sq / (2.0 * (sigma_d ** 2)))
        
        # Asymmetric Bragg Peak centered at target_local
        delta_d = depths - d_target
        sigma_depth = torch.where(delta_d < 0, self.sigma_entrance, self.sigma_distal)
        longitudinal_curve = torch.exp(- (delta_d ** 2) / (2.0 * (sigma_depth ** 2)))
        
        entrance_mask = delta_d < 0
        longitudinal_profile = torch.where(
            entrance_mask,
            self.plateau_ratio + (1.0 - self.plateau_ratio) * longitudinal_curve,
            longitudinal_curve
        )
        
        # Combine
        e_scalar = energy.view(-1)[-1]
        field = e_scalar * longitudinal_profile * lateral_profile
        
        # Ensure channel dimension: (1, X, Y, Z)
        field = field.unsqueeze(0) if field.ndim == 3 else field
        
        for key in self.keys:
            d[key] = field
            
        return d



class InjectGaussianBeamPriord(MapTransform):
    def __init__(
        self,
        keys,
        source_key="ray_source",
        target_key="ray_target",
        ref_key="ct",
        sigma=4.0,
        flip_lps_to_ras=True,
        prior_mode="to_center",   # "to_center" or "full_line" or "to_target"
        lock_z_to_center=True,
    ):
        super().__init__(keys)

        self.source_key = source_key
        self.target_key = target_key
        self.ref_key = ref_key
        self.sigma = sigma
        self.flip_lps_to_ras = flip_lps_to_ras
        self.prior_mode = prior_mode
        self.lock_z_to_center = lock_z_to_center

        valid_modes = ["to_center", "full_line", "to_target"]
        if self.prior_mode not in valid_modes:
            raise ValueError(f"prior_mode must be one of {valid_modes}, got {self.prior_mode}")

    def __call__(self, data):
        d = dict(data)

        # 1. Extract physical ray coordinates
        phys_source = np.array(d[self.source_key][:3], dtype=np.float32, copy=True)
        phys_target = np.array(d[self.target_key][:3], dtype=np.float32, copy=True)

        # 2. Coordinate system alignment: LPS -> RAS
        if self.flip_lps_to_ras:
            phys_source[0] *= -1
            phys_source[1] *= -1
            phys_target[0] *= -1
            phys_target[1] *= -1

        # 3. Physical -> voxel coordinates
        ct_tensor = d[self.ref_key]
        affine = ct_tensor.affine.cpu().numpy()
        inv_affine = np.linalg.inv(affine)

        vox_source = nib.affines.apply_affine(inv_affine, phys_source)
        vox_target = nib.affines.apply_affine(inv_affine, phys_target)

        device = ct_tensor.device

        # 4. Tensor spatial shape
        _, X, Y, Z = ct_tensor.shape

        image_center = np.array(
            [
                (X - 1) / 2.0,
                (Y - 1) / 2.0,
                (Z - 1) / 2.0,
            ],
            dtype=np.float32,
        )

        # Optional: force the ray into the middle z-slice of the crop
        if self.lock_z_to_center:
            z_center = image_center[2]
            vox_source[2] = z_center
            vox_target[2] = z_center

        # 5. Choose prior geometry mode
        if self.prior_mode == "to_center":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(image_center, dtype=torch.float32, device=device)

            # if z is locked, center is also on that z-slice
            if self.lock_z_to_center:
                B[2] = A[2]

            use_segment = True

        elif self.prior_mode == "to_target":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(vox_target, dtype=torch.float32, device=device)

            use_segment = True

        elif self.prior_mode == "full_line":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(vox_target, dtype=torch.float32, device=device)

            use_segment = False

        # 6. Generate voxel grid
        x_grid, y_grid, z_grid = torch.meshgrid(
            torch.arange(X, device=device),
            torch.arange(Y, device=device),
            torch.arange(Z, device=device),
            indexing="ij",
        )

        P = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        # 7. Distance from each voxel to line / segment
        AB = B - A
        AP = P - A

        dot_AP_AB = torch.sum(AP * AB, dim=-1)
        dot_AB_AB = torch.sum(AB * AB)

        t_raw = dot_AP_AB / (dot_AB_AB + 1e-8)

        if use_segment:
            # finite segment: A -> B
            t_used = torch.clamp(t_raw, min=0.0, max=1.0)
        else:
            # infinite line through A and B
            t_used = t_raw

        C = A + t_used.unsqueeze(-1) * AB

        dist_sq = torch.sum((P - C) ** 2, dim=-1)

        # 8. Gaussian tube
        gaussian_prior = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))

        # 9. If finite segment, zero out anything outside A -> B
        if use_segment:
            gaussian_prior[(t_raw < 0.0) | (t_raw > 1.0)] = 0.0

        d["geometric_prior"] = gaussian_prior.unsqueeze(0)

        return d

class InjectGaussianBeamPriord_old(MapTransform):
    def __init__(
        self,
        keys,
        source_key="ray_source",
        target_key="ray_target",
        ref_key="ct",
        sigma=4.0,
        flip_lps_to_ras=True,
        prior_mode="to_center",   # "to_center" or "full_line" or "to_target"
        lock_z_to_center=True,
    ):
        super().__init__(keys)

        self.source_key = source_key
        self.target_key = target_key
        self.ref_key = ref_key
        self.sigma = sigma
        self.flip_lps_to_ras = flip_lps_to_ras
        self.prior_mode = prior_mode
        self.lock_z_to_center = lock_z_to_center

        valid_modes = ["to_center", "full_line", "to_target"]
        if self.prior_mode not in valid_modes:
            raise ValueError(f"prior_mode must be one of {valid_modes}, got {self.prior_mode}")

    def __call__(self, data):
        d = dict(data)

        # 1. Extract physical ray coordinates
        phys_source = np.array(d[self.source_key][:3], dtype=np.float32, copy=True)
        phys_target = np.array(d[self.target_key][:3], dtype=np.float32, copy=True)

        # 2. Coordinate system alignment: LPS -> RAS
        if self.flip_lps_to_ras:
            phys_source[0] *= -1
            phys_source[1] *= -1
            phys_target[0] *= -1
            phys_target[1] *= -1

        # 3. Physical -> voxel coordinates
        ct_tensor = d[self.ref_key]
        dtype = ct_tensor.dtype
        device = ct_tensor.device
        if hasattr(ct_tensor, "affine"):
            affine = ct_tensor.affine.to(device, dtype)
        else:
            affine = torch.as_tensor(d.get("ct_meta_dict", {}).get("affine", torch.eye(4)), dtype=dtype, device=device)
        inv_affine = np.linalg.inv(affine)

        vox_source = nib.affines.apply_affine(inv_affine, phys_source)
        vox_target = nib.affines.apply_affine(inv_affine, phys_target)


        # 4. Tensor spatial shape
        _, X, Y, Z = ct_tensor.shape

        image_center = np.array(
            [
                (X - 1) / 2.0,
                (Y - 1) / 2.0,
                (Z - 1) / 2.0,
            ],
            dtype=np.float32,
        )

        # Optional: force the ray into the middle z-slice of the crop
        if self.lock_z_to_center:
            z_center = image_center[2]
            vox_source[2] = z_center
            vox_target[2] = z_center

        # 5. Choose prior geometry mode
        if self.prior_mode == "to_center":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(image_center, dtype=torch.float32, device=device)

            # if z is locked, center is also on that z-slice
            if self.lock_z_to_center:
                B[2] = A[2]

            use_segment = True

        elif self.prior_mode == "to_target":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(vox_target, dtype=torch.float32, device=device)

            use_segment = True

        elif self.prior_mode == "full_line":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(vox_target, dtype=torch.float32, device=device)

            use_segment = False

        # 6. Generate voxel grid
        x_grid, y_grid, z_grid = torch.meshgrid(
            torch.arange(X, device=device),
            torch.arange(Y, device=device),
            torch.arange(Z, device=device),
            indexing="ij",
        )

        P = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        # 7. Distance from each voxel to line / segment
        AB = B - A
        AP = P - A

        dot_AP_AB = torch.sum(AP * AB, dim=-1)
        dot_AB_AB = torch.sum(AB * AB)

        t_raw = dot_AP_AB / (dot_AB_AB + 1e-8)

        if use_segment:
            # finite segment: A -> B
            t_used = torch.clamp(t_raw, min=0.0, max=1.0)
        else:
            # infinite line through A and B
            t_used = t_raw

        C = A + t_used.unsqueeze(-1) * AB

        dist_sq = torch.sum((P - C) ** 2, dim=-1)

        # 8. Gaussian tube
        gaussian_prior = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))

        # 9. If finite segment, zero out anything outside A -> B
        if use_segment:
            gaussian_prior[(t_raw < 0.0) | (t_raw > 1.0)] = 0.0

        d["geometric_prior"] = gaussian_prior.unsqueeze(0)

        return d





class Ray_Info(MapTransform):
    def __init__(
        self,
        keys,
        dose_key="gt_dose",
        output_key="positions",
        z_output_key="ray_z_positions",
        threshold_rel=0.03,
        q_low=1.0,
        q_high=99.0,
        min_voxels_per_slice=3,
        normalize_by_shape=True,
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.dose_key = dose_key
        self.output_key = output_key
        self.z_output_key = z_output_key

        self.threshold_rel = threshold_rel
        self.q_low = q_low
        self.q_high = q_high
        self.min_voxels_per_slice = min_voxels_per_slice
        self.normalize_by_shape = normalize_by_shape

    def _to_numpy_3d(self, dose):
        if torch.is_tensor(dose):
            arr = dose.detach().cpu().numpy()
        else:
            arr = np.asarray(dose)

        arr = np.squeeze(arr).astype(np.float32)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D dose array after squeeze, got shape {arr.shape}")

        return arr

    def _find_beam_start_end(self, dose):
        arr = self._to_numpy_3d(dose)

        max_val = float(arr.max())

        if max_val <= 0:
            raise ValueError("Dose array is fully zero.")

        mask = arr > self.threshold_rel * max_val

        if mask.sum() < 5:
            raise ValueError(
                f"Too few voxels above threshold. "
                f"mask.sum()={mask.sum()}, threshold_rel={self.threshold_rel}."
            )

        coords = np.argwhere(mask).astype(np.float32)

        centroid = coords.mean(axis=0)
        Xc = coords - centroid

        cov = Xc.T @ Xc / max(len(coords) - 1, 1)

        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        projections = Xc @ direction

        p_low = np.percentile(projections, self.q_low)
        p_high = np.percentile(projections, self.q_high)

        point_low = centroid + p_low * direction
        point_high = centroid + p_high * direction

        def clip_round(p):
            idx = np.round(p).astype(np.int64)
            idx = np.clip(idx, [0, 0, 0], np.array(arr.shape) - 1)
            return idx

        idx_low = clip_round(point_low)
        idx_high = clip_round(point_high)

        bragg_index = np.array(
            np.unravel_index(np.argmax(arr), arr.shape),
            dtype=np.int64,
        )

        d_low_to_bragg = np.linalg.norm(idx_low - bragg_index)
        d_high_to_bragg = np.linalg.norm(idx_high - bragg_index)

        # A Bragg/max ponthoz közelebb eső vég legyen az end.
        if d_low_to_bragg < d_high_to_bragg:
            start_index = idx_high
            end_index = idx_low
        else:
            start_index = idx_low
            end_index = idx_high

        direction = end_index.astype(np.float32) - start_index.astype(np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        return {
            "start_index": start_index.astype(np.float32),
            "end_index": end_index.astype(np.float32),
            "bragg_index": bragg_index.astype(np.float32),
            "direction": direction.astype(np.float32),
            "max_dose": max_val,
            "n_mask_voxels": int(mask.sum()),
            "shape": np.array(arr.shape, dtype=np.float32),
        }

    def _find_start_end_bragg_for_each_z(self, dose):
        """
        Minden z-slice-ra kiszámolja:
            start, end, bragg

        Output shape:
            (Z, 3, 3)

        Jelentés:
            out[z, 0, :] = start [x, y, z]
            out[z, 1, :] = end   [x, y, z]
            out[z, 2, :] = bragg [x, y, z]

        Ha egy slice-on nincs elég dózis, akkor fallback:
            az egész 3D dózis alapján számolt globális start/end/bragg
            x,y koordinátáit használja, de z-t az aktuális slice-ra állítja.

        Így sehol sem lesz nullás slice.
        """

        arr = self._to_numpy_3d(dose)

        X, Y, Z = arr.shape
        max_val = float(arr.max())

        if max_val <= 0:
            raise ValueError("Dose array is fully zero.")

        threshold = self.threshold_rel * max_val

        # Globális fallback arra az esetre, ha egy slice-on kevés voxel lenne.
        global_result = self._find_beam_start_end(dose)
        global_start = global_result["start_index"]
        global_end = global_result["end_index"]
        global_bragg = global_result["bragg_index"]

        ray_z_positions = np.zeros((Z, 3, 3), dtype=np.float32)

        for z in range(Z):
            sl = arr[:, :, z]
            mask = sl > threshold

            # -----------------------------
            # Fallback, ha kevés pont van
            # -----------------------------
            if mask.sum() < self.min_voxels_per_slice:
                ray_z_positions[z, 0, :] = np.array(
                    [global_start[0], global_start[1], float(z)],
                    dtype=np.float32,
                )
                ray_z_positions[z, 1, :] = np.array(
                    [global_end[0], global_end[1], float(z)],
                    dtype=np.float32,
                )
                ray_z_positions[z, 2, :] = np.array(
                    [global_bragg[0], global_bragg[1], float(z)],
                    dtype=np.float32,
                )
                continue

            coords_xy = np.argwhere(mask).astype(np.float32)  # shape: (N, 2), [x, y]

            if len(coords_xy) < 2:
                ray_z_positions[z, 0, :] = np.array(
                    [global_start[0], global_start[1], float(z)],
                    dtype=np.float32,
                )
                ray_z_positions[z, 1, :] = np.array(
                    [global_end[0], global_end[1], float(z)],
                    dtype=np.float32,
                )
                ray_z_positions[z, 2, :] = np.array(
                    [global_bragg[0], global_bragg[1], float(z)],
                    dtype=np.float32,
                )
                continue

            # -----------------------------
            # 2D PCA az adott slice-on
            # -----------------------------
            centroid = coords_xy.mean(axis=0)
            Xc = coords_xy - centroid

            cov = Xc.T @ Xc / max(len(coords_xy) - 1, 1)

            eigvals, eigvecs = np.linalg.eigh(cov)
            direction_2d = eigvecs[:, np.argmax(eigvals)]
            direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-8)

            projections = Xc @ direction_2d

            p_low = np.percentile(projections, self.q_low)
            p_high = np.percentile(projections, self.q_high)

            point_low = centroid + p_low * direction_2d
            point_high = centroid + p_high * direction_2d

            def clip_round_2d(p):
                idx = np.round(p).astype(np.int64)
                idx = np.clip(idx, [0, 0], np.array([X, Y]) - 1)
                return idx.astype(np.float32)

            idx_low = clip_round_2d(point_low)
            idx_high = clip_round_2d(point_high)

            # Bragg/max az adott slice-on
            bragg_xy = np.array(
                np.unravel_index(np.argmax(sl), sl.shape),
                dtype=np.float32,
            )

            # A Bragg/max ponthoz közelebb eső vég legyen az end
            d_low_to_bragg = np.linalg.norm(idx_low - bragg_xy)
            d_high_to_bragg = np.linalg.norm(idx_high - bragg_xy)

            if d_low_to_bragg < d_high_to_bragg:
                start_xy = idx_high
                end_xy = idx_low
            else:
                start_xy = idx_low
                end_xy = idx_high

            start = np.array(
                [start_xy[0], start_xy[1], float(z)],
                dtype=np.float32,
            )
            end = np.array(
                [end_xy[0], end_xy[1], float(z)],
                dtype=np.float32,
            )
            bragg = np.array(
                [bragg_xy[0], bragg_xy[1], float(z)],
                dtype=np.float32,
            )

            ray_z_positions[z, 0, :] = start
            ray_z_positions[z, 1, :] = end
            ray_z_positions[z, 2, :] = bragg

        return ray_z_positions

    def __call__(self, data):
        d = dict(data)

        dose = d[self.dose_key]
        arr = self._to_numpy_3d(dose)
        shape = np.array(arr.shape, dtype=np.float32)

        result = self._find_beam_start_end(dose)

        start = result["start_index"]
        end = result["end_index"]
        bragg = result["bragg_index"]
        direction = result["direction"]

        ray_z_positions = self._find_start_end_bragg_for_each_z(dose)

        if self.normalize_by_shape:
            denom = np.maximum(shape - 1.0, 1.0)
            start_out = start / denom
            end_out = end / denom
            bragg_out = bragg / denom
            # shape: (Z, 3, 3)
            ray_z_positions_out = ray_z_positions / denom[None, None, :]
        else:
            start_out = start
            end_out = end
            bragg_out = bragg
            ray_z_positions_out = ray_z_positions

        # --- NEW 1D DISTANCE CALCULATION ---
        # Grab the ray_source (entrance) from the dictionary.
        # Note: If it's a tensor, convert to numpy for the np.linalg.norm math
        entrance = d["ray_source"]
        if isinstance(entrance, torch.Tensor):
            entrance = entrance.cpu().numpy()
            
        entrance = np.array(entrance, dtype=np.float32)

        if hasattr(dose, 'meta') and 'affine' in dose.meta:
            affine = dose.meta['affine']
        else:
            affine = d.get('affine', np.eye(4))
            
        if isinstance(affine, torch.Tensor):
            affine = affine.cpu().numpy()
        # Stack into a 3-dimensional vector for the MLP Regressor
        end_offset = end_out - start_out
        bragg_offset = bragg_out - start_out

        # Stack into a 9-dimensional vector for the Regressor
        positions = np.concatenate(
            [start_out, end_offset, bragg_offset], axis=0
        ).astype(np.float32)

        d[self.output_key] = torch.as_tensor(positions, dtype=torch.float32)

        # Debug keys for visualization
        d["ray_start_anchor"] = torch.as_tensor(start_out, dtype=torch.float32)
        d["ray_end_offset"] = torch.as_tensor(end_offset, dtype=torch.float32)
        d["ray_bragg_offset"] = torch.as_tensor(bragg_offset, dtype=torch.float32)

        return d



class InjectGaussianBeamPriordClass():
    def __init__(
        self,
        source_key="ray_source",
        target_key="ray_target",
        ref_key="ct",
        sigma=4.0,
        flip_lps_to_ras=True,
        prior_mode="to_center",   # "to_center" or "full_line" or "to_target"
        lock_z_to_center=True,
    ):
        super().__init__()

        self.source_key = source_key
        self.target_key = target_key
        self.ref_key = ref_key
        self.sigma = sigma
        self.flip_lps_to_ras = flip_lps_to_ras
        self.prior_mode = prior_mode
        self.lock_z_to_center = lock_z_to_center

        valid_modes = ["to_center", "full_line", "to_target"]
        if self.prior_mode not in valid_modes:
            raise ValueError(f"prior_mode must be one of {valid_modes}, got {self.prior_mode}")

    def __call__(self, data):
        d = dict(data)

        # 1. Extract physical ray coordinates
        phys_source = np.array(d[self.source_key][:3], dtype=np.float32, copy=True)
        phys_target = np.array(d[self.target_key][:3], dtype=np.float32, copy=True)

        # 2. Coordinate system alignment: LPS -> RAS
        if self.flip_lps_to_ras:
            phys_source[0] *= -1
            phys_source[1] *= -1
            phys_target[0] *= -1
            phys_target[1] *= -1

        # 3. Physical -> voxel coordinates
        ct_tensor = d[self.ref_key]
        affine = ct_tensor.affine.cpu().numpy()
        inv_affine = np.linalg.inv(affine)

        vox_source = nib.affines.apply_affine(inv_affine, phys_source)
        vox_target = nib.affines.apply_affine(inv_affine, phys_target)

        device = ct_tensor.device

        # 4. Tensor spatial shape
        _, X, Y, Z = ct_tensor.shape

        image_center = np.array(
            [
                (X - 1) / 2.0,
                (Y - 1) / 2.0,
                (Z - 1) / 2.0,
            ],
            dtype=np.float32,
        )

        # Optional: force the ray into the middle z-slice of the crop
        if self.lock_z_to_center:
            z_center = image_center[2]
            vox_source[2] = z_center
            vox_target[2] = z_center

        # 5. Choose prior geometry mode
        if self.prior_mode == "to_center":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(image_center, dtype=torch.float32, device=device)

            # if z is locked, center is also on that z-slice
            if self.lock_z_to_center:
                B[2] = A[2]

            use_segment = True

        elif self.prior_mode == "to_target":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(vox_target, dtype=torch.float32, device=device)

            use_segment = True

        elif self.prior_mode == "full_line":
            A = torch.tensor(vox_source, dtype=torch.float32, device=device)
            B = torch.tensor(vox_target, dtype=torch.float32, device=device)

            use_segment = False

        # 6. Generate voxel grid
        x_grid, y_grid, z_grid = torch.meshgrid(
            torch.arange(X, device=device),
            torch.arange(Y, device=device),
            torch.arange(Z, device=device),
            indexing="ij",
        )

        P = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        # 7. Distance from each voxel to line / segment
        AB = B - A
        AP = P - A

        dot_AP_AB = torch.sum(AP * AB, dim=-1)
        dot_AB_AB = torch.sum(AB * AB)

        t_raw = dot_AP_AB / (dot_AB_AB + 1e-8)

        if use_segment:
            # finite segment: A -> B
            t_used = torch.clamp(t_raw, min=0.0, max=1.0)
        else:
            # infinite line through A and B
            t_used = t_raw

        C = A + t_used.unsqueeze(-1) * AB

        dist_sq = torch.sum((P - C) ** 2, dim=-1)

        # 8. Gaussian tube
        gaussian_prior = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))

        # 9. If finite segment, zero out anything outside A -> B
        if use_segment:
            gaussian_prior[(t_raw < 0.0) | (t_raw > 1.0)] = 0.0

        return gaussian_prior.unsqueeze(0)



class InjectEnergyDepositionFieldClass():
    def __init__(
        self,
        ref_key="ct",
        source_key="ray_source",
        target_key="ray_target",
        energy_key="condition",
        spacing=(4.0, 4.0, 3.0),
        sigma_base=2.0,
        alpha=0.004,
        plateau_ratio=0.70,
        sigma_entrance=16.0,
        sigma_distal=8.0,
        allow_missing_keys=False
    ):
        super().__init__( allow_missing_keys)
        self.ref_key = ref_key
        self.source_key = source_key
        self.target_key = target_key
        self.energy_key = energy_key
        self.spacing = spacing
        self.sigma_base = sigma_base
        self.alpha = alpha
        self.plateau_ratio = plateau_ratio
        self.sigma_entrance = sigma_entrance
        self.sigma_distal = sigma_distal

    def __call__(self, data):
        d = dict(data)
        ref_img = d["ct"]
        
        if not isinstance(ref_img, torch.Tensor):
            ref_img = torch.as_tensor(ref_img)
        device = ref_img.device
        dtype = ref_img.dtype
        
        spatial_shape = ref_img.shape[-3:]
        X, Y, Z = spatial_shape
        
        # 1. Extract physical vectors and scalar energy
        source_global = torch.as_tensor(d[self.source_key], dtype=dtype, device=device)
        target_global = torch.as_tensor(d[self.target_key], dtype=dtype, device=device)
        energy = torch.as_tensor(d[self.energy_key], dtype=dtype, device=device)
        
        # 2. Calculate normalized unit beam direction vector from global points
        beam_vec_global = target_global - source_global
        d_target = torch.norm(beam_vec_global) + 1e-6
        beam_dir = beam_vec_global / d_target  # Direction vector (dx, dy, dz)
        
        # 3. Define LOCAL slab physical coordinates
        # Grid origin starts at 0 for the local cropped volume
        grid_x = torch.arange(X, device=device, dtype=dtype) * self.spacing[0]
        grid_y = torch.arange(Y, device=device, dtype=dtype) * self.spacing[1]
        grid_z = torch.arange(Z, device=device, dtype=dtype) * self.spacing[2]
        coords = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=0)
        
        # 4. Center local target at the exact middle of the cropped volume (16th slab)
        target_local = torch.tensor([
            (X * self.spacing[0]) / 2.0,
            (Y * self.spacing[1]) / 2.0,
            (Z * self.spacing[2]) / 2.0
        ], dtype=dtype, device=device)
        
        # Local source is reconstructed backwards along the beam direction
        source_local = target_local - (d_target * beam_dir)
        
        # 5. Project LOCAL coordinates onto local beam line
        v = coords - source_local.view(3, 1, 1, 1)
        depths = torch.sum(v * beam_dir.view(3, 1, 1, 1), dim=0)
        
        # 6. Lateral distance squared (r^2) from ray axis
        v_lateral = v - (depths.unsqueeze(0) * beam_dir.view(3, 1, 1, 1))
        r_sq = torch.sum(v_lateral ** 2, dim=0)
        
        # 7. Physical profiles
        depths_pos = torch.clamp(depths, min=0.0)
        sigma_d = self.sigma_base + (self.alpha * depths_pos)
        lateral_profile = torch.exp(-r_sq / (2.0 * (sigma_d ** 2)))
        
        # Asymmetric Bragg Peak centered at target_local
        delta_d = depths - d_target
        sigma_depth = torch.where(delta_d < 0, self.sigma_entrance, self.sigma_distal)
        longitudinal_curve = torch.exp(- (delta_d ** 2) / (2.0 * (sigma_depth ** 2)))
        
        entrance_mask = delta_d < 0
        longitudinal_profile = torch.where(
            entrance_mask,
            self.plateau_ratio + (1.0 - self.plateau_ratio) * longitudinal_curve,
            longitudinal_curve
        )
        
        # Combine
        e_scalar = energy.view(-1)[-1]
        field = e_scalar * longitudinal_profile * lateral_profile
        
        # Ensure channel dimension: (1, X, Y, Z)
        field = field.unsqueeze(0) if field.ndim == 3 else field
        return field
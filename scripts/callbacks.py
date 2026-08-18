import pytorch_lightning as pl
import torch
from scripts.metrics import SimpleMaskedMAE, compute_angle_agnostic_idd, GammaLoss, Stratified_plan_level_MAE
import json
import matplotlib.pyplot as plt
import wandb
import numpy as np
from monai.visualize import matshow3d

class Matshow3DVisualizerCallback(pl.Callback):
    def __init__(self, num_samples=1):
        super().__init__()
        # How many patients from the first batch you want to visualize
        self.num_samples = num_samples

        
    def shared_step(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,prefix="val"):
        # 1. SPEED SAFEGUARD: Only run this on the very first batch of the validation epoch
        if batch_idx != 0:
            return
            
        # 2. DDP SAFEGUARD: Only the main GPU (Rank 0) is allowed to plot and talk to WandB
        if not trainer.is_global_zero:
            return

        # Extract tensors from the dictionary you return in validation_step
        pred_dose = outputs["pred_dose"]
        gt_dose = outputs["gt_dose"]
        use_geom_prior = True
        use_energy_prior = True
        use_prior_extra = False

        if "geometric_prior" in batch.keys():
            use_geom_prior = True
        if "field" in batch.keys():
            use_energy_prior = True
        if "prior_extra" in outputs.keys():
            use_prior_extra = True
        
        if use_geom_prior:
            geom_prior = pl_module.generate_gaussian_prior(batch['ct'],batch['ray_source'],batch['ray_target'],batch['affine_trans']) 
            #batch["geometric_prior"]
        if use_energy_prior:
            field = pl_module.generate_energy_field(batch['ct'],source_global=batch['ray_source'],target_global=batch['ray_target'],condition=batch['condition'])
            #batch["field"]
        if use_prior_extra:
            prior_extra = outputs["prior_extra"]

        # Loop through the requested number of samples in the batch
        for i in range(min(self.num_samples, pred_dose.shape[0])):
            
            # Detach, move to CPU, and convert to numpy. 
            # Squeeze removes the channel dimension (1, 256, 256, 32) -> (256, 256, 32)
            randIdx = np.random.randint(0, pred_dose.shape[0])
            p_vol = pred_dose[randIdx].squeeze().detach().cpu().float().numpy()
            g_vol = gt_dose[randIdx].squeeze().detach().cpu().float().numpy()
            if use_geom_prior:
                prior_vol = geom_prior[randIdx].squeeze().detach().cpu().float().numpy()
            if use_energy_prior:
                prior_energy_vol = field[randIdx].squeeze().detach().cpu().float().numpy()
            if use_prior_extra:
                prior_new_vol = prior_extra[randIdx].squeeze().detach().cpu().float().numpy()
            # Calculate the Error Map (Absolute Difference)
            err_vol = np.abs(p_vol - g_vol)

            # We use a viridis colormap for dose, and inferno for the error map to make it pop
            fig_gt = plt.figure(figsize=(12, 12))
            matshow3d(g_vol, fig=fig_gt, title=f"Ground Truth Dose", cmap="viridis",frame_dim=-1)
            
            fig_pred = plt.figure(figsize=(12, 12))
            matshow3d(p_vol, fig=fig_pred, title=f"Predicted Dose", cmap="viridis",frame_dim=-1)

            fig_err = plt.figure(figsize=(12, 12))
            matshow3d(err_vol, fig=fig_err, title=f"Absolute Error Map", cmap="inferno",frame_dim=-1)
            log_dict = { f"val/visuals/Sample_{i}": [wandb.Image(fig_gt, caption=f"{prefix}/Ground Truth"),
                                                     wandb.Image(fig_pred, caption=f"{prefix}/Prediction"),
                                                    wandb.Image(fig_err, caption=f"{prefix}/Error Map")],
                         "global_step": trainer.global_step}
            if use_geom_prior:
                fig_prior = plt.figure(figsize=(12, 12))
                matshow3d(prior_vol, fig=fig_prior, title=f"Geometric Prior", cmap="viridis",frame_dim=-1)
                log_dict[f"val/visuals/Sample_{i}"].append(wandb.Image(fig_prior, caption=f"{prefix}/Geometric Prior"))

            
            if use_energy_prior:
                fig_prior_energy = plt.figure(figsize=(12, 12))
                matshow3d(prior_energy_vol, fig=fig_prior_energy, title=f"Energy Prior", cmap="viridis",frame_dim=-1)
                log_dict[f"val/visuals/Sample_{i}"].append(wandb.Image(fig_prior_energy, caption=f"{prefix}/Energy Prior"))
            if use_prior_extra:
                fig_prior_new = plt.figure(figsize=(12, 12))
                matshow3d(prior_new_vol, fig=fig_prior_new, title=f"New Prior", cmap="viridis",frame_dim=-1)
                log_dict[f"val/visuals/Sample_{i}"].append(wandb.Image(fig_prior_new, caption=f"{prefix}/New Prior"))

            # Upload the matplotlib figures directly to Weights & Biases
            trainer.logger.experiment.log(log_dict)

            # CRITICAL: Close the figures to prevent a massive RAM memory leak!
            plt.close(fig_gt)
            plt.close(fig_pred)
            plt.close(fig_err)
            if use_geom_prior:
                plt.close(fig_prior)
            if use_energy_prior:
                plt.close(fig_prior_energy)
            if use_prior_extra:
                plt.close(fig_prior_new)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.shared_step(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    #def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    #    self.shared_step(trainer, pl_module, outputs, batch, batch_idx)

class BraggPeakDistanceCallback(pl.Callback):
    def __init__(self, spacing=(4.0, 4.0, 3.0)):
        super().__init__()
        # Store the physical voxel dimensions (X, Y, Z) in mm
        self.spacing = torch.tensor(spacing)
        self.train_distance = 0.0
        self.train_samples = 0
        self.val_distance = 0.0
        self.val_samples = 0

    def _calculate_batch_distance(self, y_hat, y, device):
        # Ensure the spacing tensor is on the same GPU as the tensors
        if self.spacing.device != device:
            self.spacing = self.spacing.to(device)
            
        B, _, X, Y, Z = y_hat.shape
        
        # Flatten the spatial dimensions to find the absolute max index
        y_hat_flat = y_hat.view(B, -1)
        y_flat = y.view(B, -1)
        
        idx_pred = torch.argmax(y_hat_flat, dim=1)
        idx_true = torch.argmax(y_flat, dim=1)
        
        # Unravel the 1D indices back into 3D grid coordinates (X, Y, Z)
        x_pred = idx_pred // (Y * Z)
        rem_pred = idx_pred % (Y * Z)
        y_pred = rem_pred // Z
        z_pred = rem_pred % Z
        
        x_true = idx_true // (Y * Z)
        rem_true = idx_true % (Y * Z)
        y_true = rem_true // Z
        z_true = rem_true % Z
        
        coords_pred = torch.stack([x_pred, y_pred, z_pred], dim=1).float()
        coords_true = torch.stack([x_true, y_true, z_true], dim=1).float()
        
        # Calculate physical distance in millimeters
        physical_diff = (coords_pred - coords_true) * self.spacing
        distances = torch.linalg.norm(physical_diff, dim=1)
        
        # Return the sum of distances and the batch size
        return distances.sum().item(), B

    # ---------------------------------------------------------
    # TRAINING HOOKS
    # ---------------------------------------------------------
    #def on_train_epoch_start(self, trainer, pl_module):
    #    self.train_distance = 0.0
    #    self.train_samples = 0

    #def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #    dist_sum, b_size = self._calculate_batch_distance(outputs["pred_dose"], outputs["gt_dose"], pl_module.device)
    #    self.train_distance += dist_sum
    #    self.train_samples += b_size

    #def on_train_epoch_end(self, trainer, pl_module):
    #    if self.train_samples > 0:
    #        mean_distance_mm = self.train_distance / self.train_samples
    #        pl_module.log("train/bragg_peak_error_mm", mean_distance_mm, sync_dist=True)

    # ---------------------------------------------------------
    # VALIDATION HOOKS
    # ---------------------------------------------------------
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_distance = 0.0
        self.val_samples = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        dist_sum, b_size = self._calculate_batch_distance(outputs["pred_dose"], outputs["gt_dose"], pl_module.device)
        self.val_distance += dist_sum
        self.val_samples += b_size

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_samples > 0:
            mean_distance_mm = self.val_distance / self.val_samples
            pl_module.log("val/bragg_peak_error_mm", mean_distance_mm, sync_dist=True)

import torch.nn.functional as F

class DoseLevel1MetricsCallback(pl.Callback):
    def __init__(self,voxel_spacing=(4.0, 4.0, 3.0)):
        super().__init__() # Good practice to init the parent class
        self.beam_masked_mae_loss = SimpleMaskedMAE()
        self.compute_angle_agnostic_idd = compute_angle_agnostic_idd
        self.GammaLoss = GammaLoss()
        self.Stratified_plan_level_MAE = Stratified_plan_level_MAE()
        self.voxel_spacing = voxel_spacing

        
    def shared_step(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,prefix="val"):
        # Extract tensors from the validation_step outputs
        gt_dose = batch["gt_dose"]
        pred_dose = outputs["pred_dose"]
        ray_source = batch["ray_source"]
        ray_target = batch["ray_target"]

        # Compute the beam-level losses
        beam_masked_mae = self.beam_masked_mae_loss(pred_dose, gt_dose)
        pred_idd = self.compute_angle_agnostic_idd(pred_dose, ray_source, ray_target, spacing=self.voxel_spacing)
        target_idd = self.compute_angle_agnostic_idd(gt_dose, ray_source, ray_target, spacing=self.voxel_spacing)
        idd_curve_loss_value = F.l1_loss(pred_idd, target_idd)
        gamma_loss_value = self.GammaLoss(pred_dose, gt_dose)
        stratified_mae_value = self.Stratified_plan_level_MAE(pred_dose, gt_dose)

        # Log the losses with sync_dist=True to safely average across all GPUs
        pl_module.log(f"{prefix}/beam_masked_mae", beam_masked_mae.detach(), 
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        pl_module.log(f"{prefix}/idd_curve_loss", idd_curve_loss_value.detach(), 
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log(f"{prefix}/gamma_loss", gamma_loss_value.detach(), 
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log(f"{prefix}/stratified_mae", stratified_mae_value.detach(), 
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.shared_step(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, prefix="val")
    #def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    #    self.shared_step(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, prefix="train")

import pytorch_lightning as pl
import torch
from scripts.metrics import Stratified_plan_level_MAE, GammaLoss

class DoseLevel2MetricsCallback(pl.Callback):
    def __init__(self, unique_patient_ids, volume_shape=(1, 256, 256, 32)):
        super().__init__()
        self.stratified_mae_loss = Stratified_plan_level_MAE()
        self.gamma_loss = GammaLoss()
        
        # We need all GPUs to iterate over the exact same patient list.
        # If one GPU skips a patient during a network sync, the whole script deadlocks.
        self.unique_patient_ids = unique_patient_ids
        self.volume_shape = volume_shape
        
        self.plan_preds = {}
        self.plan_gts = {}

    def on_validation_epoch_start(self, trainer, pl_module):
        # Initialize empty canvases for EVERY patient on EVERY GPU on the CPU.
        # This guarantees that when we run all_reduce later, no GPU misses a sync barrier.
        for pid in self.unique_patient_ids:
            self.plan_preds[pid] = torch.zeros(self.volume_shape, dtype=torch.float32, device='cpu')
            self.plan_gts[pid] = torch.zeros(self.volume_shape, dtype=torch.float32, device='cpu')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        gt_dose = outputs["gt_dose"]
        pred_dose = outputs["pred_dose"]
        patient_ids = outputs["patient_id"]

        for i, pid in enumerate(patient_ids):
            # Accumulate locally on the CPU (prevents VRAM Out-of-Memory crashes)
            self.plan_preds[pid] += pred_dose[i].detach()
            self.plan_gts[pid] += gt_dose[i].detach()

    def on_validation_epoch_end(self, trainer, pl_module):
        total_gamma = 0.0
        total_stratified_mae = 0.0
        
        device = pl_module.device

        for pid in self.unique_patient_ids:
            # 1. Move the local CPU sum back to the GPU
            local_pred = self.plan_preds[pid].to(device)
            local_gt = self.plan_gts[pid].to(device)

            # 2. Network Sync: Sum the partial volumes across all 4 A100 GPUs!
            full_pred = trainer.strategy.reduce(local_pred, reduce_op="sum")
            full_gt = trainer.strategy.reduce(local_gt, reduce_op="sum")

            # 3. Calculate metrics ONLY on the fully assembled patient volumes
            # Gamma expects 3D shapes (D, H, W). We squeeze out the channel dimension.
            gamma_pass_rate = self.gamma_loss.local_gamma_3d(full_pred.squeeze(), full_gt.squeeze())
            strat_mae = self.stratified_mae_loss(full_pred, full_gt)

            total_gamma += gamma_pass_rate
            total_stratified_mae += strat_mae

        # Average across the number of patients in the validation set
        avg_gamma = total_gamma / len(self.unique_patient_ids)
        avg_strat_mae = total_stratified_mae / len(self.unique_patient_ids)

        # Log the final metrics
        pl_module.log("val/Level2_GammaPassRate", avg_gamma.detach(), sync_dist=True)
        pl_module.log("val/Level2_StratifiedMAE", avg_strat_mae.detach(), sync_dist=True)

        # Free memory before the next training epoch begins
        self.plan_preds.clear()
        self.plan_gts.clear()


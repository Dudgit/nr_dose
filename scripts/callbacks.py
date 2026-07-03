import pytorch_lightning as pl
import torch
from scripts.metrics import BeamMaskedMAELoss, IDDCurveLoss, Stratified_plan_level_MAE, GammaLoss
import json

class DoseLevel1MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__() # Good practice to init the parent class
        self.beam_masked_mae_loss = BeamMaskedMAELoss()
        self.idd_curve_loss = IDDCurveLoss()
        
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Extract tensors from the validation_step outputs
        gt_dose = outputs["gt_dose"]
        pred_dose = outputs["pred_dose"]

        # Compute the beam-level losses
        beam_masked_mae = self.beam_masked_mae_loss(pred_dose, gt_dose)
        idd_curve_loss_value = self.idd_curve_loss(pred_dose, gt_dose)

        # Log the losses with sync_dist=True to safely average across all GPUs
        pl_module.log("val/beam_masked_mae", beam_masked_mae, 
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        pl_module.log("val/idd_curve_loss", idd_curve_loss_value, 
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        

import pytorch_lightning as pl
import torch
from scipts.metrics import Stratified_plan_level_MAE, GammaLoss

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
            self.plan_preds[pid] += pred_dose[i].detach().cpu()
            self.plan_gts[pid] += gt_dose[i].detach().cpu()

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
        pl_module.log("val/Level2_GammaPassRate", avg_gamma, sync_dist=True)
        pl_module.log("val/Level2_StratifiedMAE", avg_strat_mae, sync_dist=True)

        # Free memory before the next training epoch begins
        self.plan_preds.clear()
        self.plan_gts.clear()
import pytorch_lightning as pl
import torch
from scripts.metrics import BeamMaskedMAELoss, IDDCurveLoss
import torch.nn as nn

class DoseTrainer(pl.LightningModule):
    def __init__(self,model,loss_function=None,lr = 1e-4,use_warmups=False,useEnergyPrior=True,use_maskPrediction=True):
        super().__init__()
        self.model = model
        self.lr = lr
        self.use_wamrups = use_warmups
        self.loss_function = loss_function if loss_function is not None else torch.nn.L1Loss()
        self.useEnergyPrior = useEnergyPrior
    def forward(self, x,condition):
        return self.model(x,condition)
    
    def shared_step(self, batch,prefix="Train"):
        x = batch['ct']
        y = batch['gt_dose']
        ray_source = batch['ray_source']
        ray_target = batch['ray_target']
        condition = batch['condition']
        orig_shape = batch['orig_shape']

        prior = self.generate_gaussian_prior(x,phys_source=ray_source,phys_target=ray_target,affine_trans=batch['affine_trans'],z_start=batch['z_start'],orig_shape=orig_shape)
        prior_extra = self.generate_energy_field(x,phys_source=ray_source,phys_target=ray_target,condition=condition,z_start=batch['z_start'],affine_trans=batch['affine_trans'],orig_shape=orig_shape)
        x = torch.cat([x, prior, prior_extra], dim=1) if self.useEnergyPrior else torch.cat([x, prior], dim=1)  # 64x1x128x128x32 -> 64x3x128x128x32
        y_hat = self(x, condition)

        
        use_bragg_peak_loss = self.current_epoch >= 50
        use_idd_loss = self.current_epoch >= 50
        loss_dict = self.loss_function(y_hat, y, ray_source, ray_target, use_bragg_peak_loss=use_bragg_peak_loss,use_idd_loss = use_idd_loss)

        loss = loss_dict["total_loss"]
        self.logging_step(loss_dict, prefix)
        return loss, y_hat, y, condition
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.shared_step(batch,prefix="train")
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y, condition = self.shared_step(batch,prefix="val")
        return {"loss": loss.detach(), "pred_dose": y_hat.detach(), "gt_dose": y.detach(), "condition": condition.detach()}

    def logging_step(self,res_dict,prefix):
        for k,v in res_dict.items():
            self.log(f"{prefix}/{k}",v.detach(),prog_bar=True,sync_dist=True)

    def predict(self, batch):
        x = batch['ct']
        prior = batch['geometric_prior']
        prior_extra = batch['field']
        x = torch.cat([x, prior, prior_extra], dim=1)  # 64x1x128x128x32 -> 64x3x128x128x32
        condition = batch['condition']
        y_hat = self(x, condition)

        return y_hat

    def generate_gaussian_prior(self, ct_tensor, phys_source, phys_target, affine_trans, z_start, orig_shape):
        device = ct_tensor.device
        dtype = ct_tensor.dtype
        B, C, X, Y, Z = ct_tensor.shape
        
        # Flip LPS -> RAS batched
        phys_source = phys_source.clone()
        phys_target = phys_target.clone()
        phys_source[:, 0:2] *= -1
        phys_target[:, 0:2] *= -1
        
        homo_source = torch.cat([phys_source, torch.ones(B, 1, device=device)], dim=1).unsqueeze(2)
        homo_target = torch.cat([phys_target, torch.ones(B, 1, device=device)], dim=1).unsqueeze(2)
        
        inv_affine = torch.linalg.inv(affine_trans.float())
        A = torch.bmm(inv_affine, homo_source).squeeze(2)[:, :3]
        B_target = torch.bmm(inv_affine, homo_target).squeeze(2)[:, :3]
        
        # Calculate X/Y center-crop offsets
        x_offset = torch.div((orig_shape[:, 0] - X), 2, rounding_mode='floor')
        y_offset = torch.div((orig_shape[:, 1] - Y), 2, rounding_mode='floor')

        # Apply shifts exactly once
        A[:, 0] -= x_offset
        A[:, 1] -= y_offset
        A[:, 2] -= z_start.view(-1)

        B_target[:, 0] -= x_offset
        B_target[:, 1] -= y_offset
        B_target[:, 2] -= z_start.view(-1)
        
        # Generate Grid
        x_grid = torch.arange(X, device=device, dtype=dtype)
        y_grid = torch.arange(Y, device=device, dtype=dtype)
        z_grid = torch.arange(Z, device=device, dtype=dtype)
        grid_x, grid_y, grid_z = torch.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
        
        P = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        A_expanded = A.view(B, 3, 1, 1, 1)
        B_expanded = B_target.view(B, 3, 1, 1, 1)
        
        AB = B_expanded - A_expanded
        AP = P - A_expanded
        
        dot_AP_AB = torch.sum(AP * AB, dim=1)
        dot_AB_AB = torch.sum(AB * AB, dim=1)
        
        t_raw = dot_AP_AB / (dot_AB_AB + 1e-8)
        C = A_expanded + t_raw.unsqueeze(1) * AB
        dist_sq = torch.sum((P - C) ** 2, dim=1)
        
        sigma = 4.0
        gaussian_prior = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        
        return gaussian_prior.unsqueeze(1)


    def generate_energy_field(self, ct_tensor, phys_source, phys_target, condition, affine_trans, z_start,orig_shape):
        device = ct_tensor.device
        dtype = ct_tensor.dtype
        B, C, X, Y, Z = ct_tensor.shape
        spacing = torch.tensor([1.0, 1.0, 3.0], device=device, dtype=dtype).view(1, 3)
        
        # 1. Get Local Voxel Coordinates (Exactly like the Gaussian Prior!)
        phys_source_clone = phys_source.clone()
        phys_target_clone = phys_target.clone()
        phys_source_clone[:, 0:2] *= -1
        phys_target_clone[:, 0:2] *= -1
        
        homo_source = torch.cat([phys_source_clone, torch.ones(B, 1, device=device)], dim=1).unsqueeze(2)
        homo_target = torch.cat([phys_target_clone, torch.ones(B, 1, device=device)], dim=1).unsqueeze(2)
        
        inv_affine = torch.linalg.inv(affine_trans.float())
        A_vox = torch.bmm(inv_affine, homo_source).squeeze(2)[:, :3]
        B_vox = torch.bmm(inv_affine, homo_target).squeeze(2)[:, :3]
        x_offset = torch.div((orig_shape[:, 0] - X), 2, rounding_mode='floor')
        y_offset = torch.div((orig_shape[:, 1] - Y), 2, rounding_mode='floor')
        
        # Shift Global Z to Local Crop Z
        A_vox[:, 0] -= x_offset
        A_vox[:, 1] -= y_offset
        A_vox[:, 2] -= z_start.view(-1)
        
        # 2. Convert Local Voxels to Local Physical mm (for dose curve math)
        B_vox[:, 0] -= x_offset
        B_vox[:, 1] -= y_offset
        B_vox[:, 2] -= z_start.view(-1)

        A_phys = A_vox * spacing
        B_phys = B_vox * spacing
        
        # 3. Create Local Physical Grid
        grid_x = torch.arange(X, device=device, dtype=dtype) * spacing[0, 0]
        grid_y = torch.arange(Y, device=device, dtype=dtype) * spacing[0, 1]
        grid_z = torch.arange(Z, device=device, dtype=dtype) * spacing[0, 2]
        coords = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=0) 
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1, -1) # [Batch, 3, X, Y, Z]
        
        # 4. Beam Vectors in Local Physical Space
        beam_vec = B_phys - A_phys
        d_target = torch.norm(beam_vec, dim=1, keepdim=True) + 1e-6
        beam_dir = beam_vec / d_target  
        
        A_phys_view = A_phys.view(B, 3, 1, 1, 1)
        beam_dir_view = beam_dir.view(B, 3, 1, 1, 1)
        d_target_view = d_target.view(B, 1, 1, 1, 1)
        
        # 5. Project grid onto beam vector
        v = coords - A_phys_view
        depths = torch.sum(v * beam_dir_view, dim=1)  # Depth along beam
        
        v_lateral = v - (depths.unsqueeze(1) * beam_dir_view)
        r_sq = torch.sum(v_lateral ** 2, dim=1)  # Lateral radius squared
        
        # 6. Apply Physics equations
        depths_pos = torch.clamp(depths, min=0.0)
        
        sigma_base = 2.0
        alpha = 0.004
        plateau = 0.7
        sig_ent = 16.0
        sig_dist = 8.0
        
        sigma_d = sigma_base + (alpha * depths_pos)
        lateral_profile = torch.exp(-r_sq / (2.0 * (sigma_d ** 2)))
        
        delta_d = depths - d_target_view.squeeze(1) # Distance from Bragg peak target
        sigma_depth = torch.where(delta_d < 0, sig_ent, sig_dist)
        
        longitudinal_curve = torch.exp(- (delta_d ** 2) / (2.0 * (sigma_depth ** 2)))
        entrance_mask = delta_d < 0
        longitudinal_profile = torch.where(
            entrance_mask,
            plateau + (1.0 - plateau) * longitudinal_curve,
            longitudinal_curve
        )
        
        e_scalar = condition[:, 2].view(B, 1, 1, 1) 
        field = e_scalar * longitudinal_profile * lateral_profile
        
        return field.unsqueeze(1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5,fused=True)
        if self.use_wamrups:
            total_steps = self.trainer.estimated_stepping_batches
            max_epochs = self.trainer.max_epochs
            steps_per_epoch = total_steps // max_epochs
            warmup_steps = steps_per_epoch * 3
            
            # 1. Warmup: Start at 1/1000th of 1e-3, scale linearly to exactly 1e-3 over warmup_steps
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,end_factor=1.0, total_iters=warmup_steps)
            flat_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=(total_steps - warmup_steps))
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_scheduler, flat_scheduler],milestones=[warmup_steps])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", # Steps after every batch
                },
        }
        return optimizer
    

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class DoseGANTrainer(pl.LightningModule):
    def __init__(self, generator, discriminator, loss_function=None, adv_weight=0.1,d_update_freq=1,lr = 1e-4,
                 start_epoch=50,ramp_length=50, max_mae_weight=10.0,def_mae_weight=10.0, useRamping=False,earlyRamp=False,use = True):
        super().__init__()
        self.model = generator
        self.discriminator = discriminator
        self.loss_function = loss_function if loss_function is not None else torch.nn.L1Loss()
        self.adv_weight = adv_weight
        self.lr = lr
        self.automatic_optimization = False
        self.d_update_freq = d_update_freq 
        self.start_epoch = start_epoch
        self.ramp_length = ramp_length
        self.max_mae_weight = max_mae_weight
        self.def_mae_weight = def_mae_weight
        self.useRamping = useRamping
        self.earlyRamp = earlyRamp
    def forward(self, x, condition):
        return self.model(x, condition)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr, weight_decay=1e-5)
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        x = batch['ct']
        prior = batch['geometric_prior']
        x = torch.cat([x, prior], dim=1)  # Concatenate along the channel dimension
        y = batch['gt_dose']
        condition = batch['condition']
        ray_source = batch['ray_source']
        ray_target = batch['ray_target']
        
        opt_g, opt_d = self.optimizers()

        # ==========================================
        # PHASE 1: TRAIN DISCRIMINATOR
        # ==========================================
        y_hat = self.model(x, condition)
        
        # Real Pair
        real_pair = torch.cat([x, y], dim=1)
        d_real_logits = self.discriminator(real_pair)[-1]

        smoothed_real_labels = torch.empty_like(d_real_logits).fill_(0.9)
        d_loss_real = F.binary_cross_entropy_with_logits(d_real_logits, smoothed_real_labels)


        #d_loss_real = F.binary_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits))
        
        # Fake Pair (Detached)
        fake_pair = torch.cat([x, y_hat.detach()], dim=1)
        d_fake_logits = self.discriminator(fake_pair)[-1]
        d_loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        if batch_idx % self.d_update_freq == 0:  
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

        # --- Calculate D Probability Ratios ---
        with torch.no_grad():
            d_prob_real = torch.sigmoid(d_real_logits).mean()
            d_prob_fake = torch.sigmoid(d_fake_logits).mean()


        ramp_cond = self.current_epoch < self.start_epoch if self.earlyRamp else self.current_epoch > self.start_epoch
        if ramp_cond and self.useRamping:
            ram_progress = min(1.0, (self.current_epoch-self.start_epoch)/self.ramp_length)
            self.loss_function.masked_factor = self.def_mae_weight+ram_progress*(self.max_mae_weight-self.def_mae_weight)
        is_finetune = self.current_epoch > 100
        loss_dict = self.loss_function(y_hat, y, ray_source, ray_target,fine_tune=is_finetune)
        physics_loss = loss_dict["total_loss"]
        
        # Evaluate generator against updated discriminator
        fake_pair_for_g = torch.cat([x, y_hat], dim=1)
        g_fake_logits = self.discriminator(fake_pair_for_g)[-1]
        g_adv_loss = F.binary_cross_entropy_with_logits(g_fake_logits, torch.ones_like(g_fake_logits))
        
        g_loss = physics_loss + (self.adv_weight * g_adv_loss)
        loss_dict['g_adv_loss'] = g_adv_loss
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # --- Calculate G Probability & Balance Ratios ---
        with torch.no_grad():
            g_prob_fake = torch.sigmoid(g_fake_logits).mean()
            # Ratio of adversarial pressures (avoid devision by zero)
            adv_ratio = (self.adv_weight * g_adv_loss) / (d_loss + 1e-8)
            
        metrics = {
            # Probability Ratios (Crucial for stability check!)
            "ratio/d_prob_real": d_prob_real,   # Should hover 0.5 - 0.8
            "ratio/d_prob_fake": d_prob_fake,   # Should hover 0.2 - 0.5
            "ratio/g_prob_fake": g_prob_fake,   # Should hover 0.3 - 0.6
            
            # Balance Diagnostic
            "ratio/adv_to_d_loss": adv_ratio,
        }
        self.logging_step(loss_dict, prefix="train")
        for k, v in metrics.items():
            self.log(k, v, prog_bar=False, sync_dist=True)
            
        # Keep main metrics on progress bar
        self.log("D_prob_real", d_prob_real, prog_bar=True)
        self.log("G_prob_fake", g_prob_fake, prog_bar=True)
        return {"loss": g_loss, "pred_dose": y_hat, "gt_dose": y, "condition": condition}

    def validation_step(self, batch, batch_idx):
        x, y, condition = batch['ct'], batch['gt_dose'], batch['condition']
        x = torch.cat([x, batch['geometric_prior']], dim=1)  # Concatenate along the channel dimension
        y_hat = self(x, condition)
        is_finetune = self.current_epoch > 100
        loss_dict = self.loss_function(y_hat, y, batch['ray_source'], batch['ray_target'], fine_tune=is_finetune)
        self.logging_step(loss_dict,prefix ="val")
        return {"loss": loss_dict["total_loss"], "pred_dose": y_hat, "gt_dose": y, "condition": condition}
    
    def logging_step(self,res_dict,prefix):
        for k,v in res_dict.items():
            self.log(f"{prefix}/{k}",v,prog_bar=True,sync_dist=True)


class GT_Posed(pl.LightningModule):
    def __init__(self, model, loss_function=None, lr=1e-4,priorCreator=None,useEnergyPrior=True,useGeometricPrior=True,use_maskPrediction=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.PriorCreator = priorCreator
        self.loss_function = loss_function if loss_function is not None else torch.nn.L1Loss()
        self.loss_coord = torch.nn.L1Loss()
        self.useEnergyPrior = useEnergyPrior
        self.useGeometricPrior = useGeometricPrior
        self.use_maskPrediction = use_maskPrediction
        self.useNewPrior = True

    def forward(self, x,condition):
            return self.model(x,condition)

    def shared_step(self,batch,prefix="train"):
        x = batch['ct']
        y = batch['gt_dose']
        ray_source = batch['ray_source']
        ray_target = batch['ray_target']
        geom_prior = batch['geometric_prior']
        field = batch['field']
        condition = batch['condition']

            
        start_anchor = batch['ray_start_anchor']
        end_offset   = batch['ray_end_offset']
        bragg_offset = batch['ray_bragg_offset']

        # 2. Reconstruct the coordinates 
        beam_start = start_anchor
        beam_end   = start_anchor + end_offset
        bragg_peak = start_anchor + bragg_offset

        prior_extra = self.PriorCreator(batch['ct'], beam_start, beam_end,bragg_peak,condition)

        x = torch.cat([x, prior_extra], dim=1)

        if self.useEnergyPrior:
            x = torch.cat([x, field], dim=1)
        if self.useGeometricPrior:
            x = torch.cat([x, geom_prior], dim=1)
        

        y_hat = self(x, condition)
        if self.use_maskPrediction:
            mask = (geom_prior > 1e-4).float()
            y_hat = y_hat * mask

        use_bragg_peak_loss = self.current_epoch >= 50
        loss_dict = self.loss_function(y_hat, y, ray_source, ray_target, use_bragg_peak_loss=use_bragg_peak_loss)
        loss = loss_dict["total_loss"]
        self.logging_step(loss_dict, prefix)
        
        return loss, y_hat, y, condition, prior_extra

    def training_step(self, batch, batch_idx):
        loss, y_hat, y, condition, prior_extra = self.shared_step(batch,prefix="train")
        return {"loss": loss, "pred_dose": y_hat, "gt_dose": y, "condition": condition,"prior_extra":prior_extra}

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y, condition, prior_extra = self.shared_step(batch,prefix="val")
        return {"loss": loss, "pred_dose": y_hat, "gt_dose": y, "condition": condition,"prior_extra":prior_extra}

    def logging_step(self,res_dict,prefix):
        for k,v in res_dict.items():
            self.log(f"{prefix}/{k}",v,prog_bar=True,sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

        


class DBTrainer(pl.LightningModule):
    def __init__(self, encoder, decoder, transformer, latent_regressor, loss_function=None, lr=1e-4, useEnergyPrior=True, useGeometricPrior=True, useGTPosInLearning=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transformer = transformer
        self.latent_regressor = latent_regressor
        #self.film_block = filmBlock  # Fixed: changed from self.model
        self.lr = lr
        
        # Fixed: changed from self.loss_pred so it matches shared_step
        self.loss_function = loss_function if loss_function is not None else torch.nn.L1Loss()
        self.loss_coord = torch.nn.L1Loss()
        
        self.useEnergyPrior = useEnergyPrior
        self.useGeometricPrior = useGeometricPrior
        self.useGTPosInLearning = useGTPosInLearning

    def forward(self, x, condition, gt_positions=None):
        # 1. Encode and get global context
        latent = self.encoder(x)
        latent = self.transformer(latent, condition)
        pred_pos = self.latent_regressor(latent)
        film_condition = gt_positions if gt_positions is not None else pred_pos
        pred_dose = self.decoder(latent, film_condition)
        return pred_dose, pred_pos
    
    def shared_step(self, batch, prefix="Train"):
        x = batch['ct']
        y = batch['gt_dose']
        ray_source = batch['ray_source']
        ray_target = batch['ray_target']
        gt_positions = batch['positions']
        condition = batch['condition']

        # Concat Priors
        if self.useGeometricPrior:
            prior = batch['geometric_prior']
            x = torch.cat([x, prior], dim=1)  
        if self.useEnergyPrior:
            prior_extra = batch['field']
            x = torch.cat([x, prior_extra], dim=1)  

        if self.useGTPosInLearning and self.training:
            y_hat, pred_pos = self(x, condition, gt_positions)
        else:
            y_hat, pred_pos = self(x, condition)
        
        use_bragg_peak_loss = self.current_epoch >= 50
        
        # Calculate main dose loss (using your custom loss function)
        loss_dict = self.loss_function(y_hat, y, ray_source, ray_target, use_bragg_peak_loss=use_bragg_peak_loss)
        loss = loss_dict["total_loss"]

        # Calculate and add Coordinate Loss
        pos_loss = self.loss_coord(pred_pos, gt_positions)
        loss_dict["position_loss"] = pos_loss
        loss += pos_loss
        
        self.logging_step(loss_dict, prefix)
        
        return loss, y_hat, y, condition

    def training_step(self, batch, batch_idx):
        loss, y_hat, y, condition = self.shared_step(batch,prefix="train")
        
        return {"loss": loss, "pred_dose": y_hat, "gt_dose": y, "condition": condition}

    def validation_step(self, batch, batch_idx):
            loss, y_hat, y, condition = self.shared_step(batch,prefix="val")
            return {"loss": loss, "pred_dose": y_hat, "gt_dose": y, "condition": condition}
    
    def logging_step(self,res_dict,prefix):
        for k,v in res_dict.items():
            self.log(f"{prefix}/{k}",v,prog_bar=True,sync_dist=True)

    def predict(self, batch):
        x = batch['ct']
        prior = batch['geometric_prior']
        prior_extra = batch['field']
        condition = batch['condition']

        # Concat Priors
        if self.useGeometricPrior:
            x = torch.cat([x, prior], dim=1)  
        if self.useEnergyPrior:
            x = torch.cat([x, prior_extra], dim=1)  
        y_hat, pred_pos = self(x, condition)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer
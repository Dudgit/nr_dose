import pytorch_lightning as pl
import torch
from scripts.metrics import BeamMaskedMAELoss, IDDCurveLoss
import torch.nn as nn



class DoseTrainer(pl.LightningModule):
    def __init__(self,model,loss_function=None):
        super().__init__()
        self.model = model
        self.loss_function = loss_function if loss_function is not None else torch.nn.L1Loss()
    
    def forward(self, x,condition):
        return self.model(x,condition)
    
    def shared_step(self, batch,prefix="Train"):
        x = batch['ct']
        y = batch['gt_dose']
        condition = batch['condition']
        y_hat = self(x, condition)
        
        loss_dict = self.loss_function(y_hat, y)
        loss = loss_dict["total_loss"]
        self.logging_step(loss_dict, prefix)
        return loss, y_hat, y, condition
    
    def training_step(self, batch, batch_idx):
        loss, y_hat, y, condition = self.shared_step(batch,prefix="train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y, condition = self.shared_step(batch,prefix="val")
        self.log("val/MSE", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss, "pred_dose": y_hat, "gt_dose": y, "condition": condition}

    def logging_step(self,res_dict,prefix):
        for k,v in res_dict.items():
            self.log(f"{prefix}/{k}",v,prog_bar=True,sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer
    

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class DoseGANTrainer(pl.LightningModule):
    def __init__(self, generator, discriminator, loss_function=None, adv_weight=0.1,d_update_freq=1):
        super().__init__()
        self.model = generator
        self.discriminator = discriminator
        self.loss_function = loss_function if loss_function is not None else torch.nn.L1Loss()
        self.adv_weight = adv_weight
        self.automatic_optimization = False
        self.d_update_freq = d_update_freq  

    def forward(self, x, condition):
        return self.model(x, condition)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, weight_decay=1e-5)
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        x = batch['ct']
        prior = batch['geometric_prior']
        x = torch.cat([x, prior], dim=1)  # Concatenate along the channel dimension
        y = batch['gt_dose']
        condition = batch['condition']
        
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

        # ==========================================
        # PHASE 2: TRAIN GENERATOR
        # ==========================================
        loss_dict = self.loss_function(y_hat, y)
        physics_loss = loss_dict["total_loss"]
        
        # Evaluate generator against updated discriminator
        fake_pair_for_g = torch.cat([x, y_hat], dim=1)
        g_fake_logits = self.discriminator(fake_pair_for_g)[-1]
        g_adv_loss = F.binary_cross_entropy_with_logits(g_fake_logits, torch.ones_like(g_fake_logits))
        
        g_loss = physics_loss + (self.adv_weight * g_adv_loss)
        
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # --- Calculate G Probability & Balance Ratios ---
        with torch.no_grad():
            g_prob_fake = torch.sigmoid(g_fake_logits).mean()
            # Ratio of adversarial pressures (avoid devision by zero)
            adv_ratio = (self.adv_weight * g_adv_loss) / (d_loss + 1e-8)

        # ==========================================
        # DIAGNOSTIC LOGGING TO WANDB / TENSORBOARD
        # ==========================================
        metrics = {
            # Standard Losses
            "loss/physics_mae": physics_loss,
            "loss/d_total": d_loss,
            "loss/g_adv": g_adv_loss,
            "loss/g_total": g_loss,
            
            # Probability Ratios (Crucial for stability check!)
            "ratio/d_prob_real": d_prob_real,   # Should hover 0.5 - 0.8
            "ratio/d_prob_fake": d_prob_fake,   # Should hover 0.2 - 0.5
            "ratio/g_prob_fake": g_prob_fake,   # Should hover 0.3 - 0.6
            
            # Balance Diagnostic
            "ratio/adv_to_d_loss": adv_ratio,
        }
        
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
        loss_dict = self.loss_function(y_hat, y)
        self.log("val/MSE", loss_dict["total_loss"], prog_bar=True, sync_dist=True)
        return {"loss": loss_dict["total_loss"], "pred_dose": y_hat, "gt_dose": y, "condition": condition}
    
    def logging_step(self,res_dict,prefix):
        for k,v in res_dict.items():
            self.log(f"{prefix}/{k}",v,prog_bar=True,sync_dist=True)

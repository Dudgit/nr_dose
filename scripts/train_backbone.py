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
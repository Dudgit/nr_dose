import pytorch_lightning as pl
import torch
from scripts.metrics import BeamMaskedMAELoss, IDDCurveLoss

class Level1LossFunction(pl.LightningModule):
    def __init__(self,masked_factor=0.5,iid_curve_weight=0.5, allMAE_weight=0.5):
        super().__init__()
        self.beam_masked_mae_loss = BeamMaskedMAELoss()
        self.IID_curve_loss = IDDCurveLoss()
        self.allMAE = torch.nn.L1Loss()
        self.masked_factor = masked_factor
        self.iid_curve_weight = iid_curve_weight
        self.allMAE_weight = allMAE_weight
    
    def __call__(self, pred_dose, gt_dose):
        beam_masked_mae = self.beam_masked_mae_loss(pred_dose, gt_dose)
        idd_curve_loss_value = self.IID_curve_loss(pred_dose, gt_dose)
        allMAE = self.allMAE(pred_dose, gt_dose)
        total_loss = beam_masked_mae * self.masked_factor + idd_curve_loss_value * self.iid_curve_weight + allMAE * self.allMAE_weight
        lossDict = {
            "beam_masked_mae": beam_masked_mae,
            "idd_curve_loss": idd_curve_loss_value,
            "allMAE": allMAE,
            "total_loss": total_loss
        }
        return lossDict



class DoseTrainer(pl.LightningModule):
    def __init__(self,model,loss_function=None):
        super().__init__()
        self.model = model
        self.loss_function = loss_function if loss_function is not None else Level1LossFunction()
    
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
        loss, y_hat, y, condition = self.shared_step(batch,prefix="Train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y, condition = self.shared_step(batch,prefix="Val")
        self.log("val/MSE", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss, "pred_dose": y_hat, "gt_dose": y, "condition": condition}

    def logging_step(self,res_dict,prefix):
        for k,v in res_dict.items():
            self.log(f"{prefix}/{k}",v,prog_bar=True,sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer
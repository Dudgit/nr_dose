import torch
import pytorch_lightning as pl



class DoseTrainer(pl.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model = model

    def init_loss(self,lr,loss_function):
        self.lr = lr
        self.loss_function = loss_function if loss_function is not None else torch.nn.L1Loss()
    
    def forward(self, x,condition):
        return self.model(x,condition)
    
    def predict(self, batch):
        x = batch['ct']
        energy = batch['energy']
        y = batch['dose']
        prior = self.generate_prior(ct = x, phys_source = batch['beam_source'], phys_target = batch['beam_target'], 
                                    affine_trans = batch['ct_meta_dict']['affine'], energy = energy)
        x = torch.cat([x, prior], dim=1)  # 64x1x128x128x32 -> 64x3x128x128x32
        y_hat = self(x, energy)
 
        return y_hat, y

    def shared_step(self, batch,prefix="Train"):
        y_hat, y = self.predict(batch)
        loss_dict = self.loss_function(y_hat, y) # Modify according to lossWrapper

        loss = loss_dict["total_loss"]
        self.logging_step(loss_dict, prefix)
        return loss, y_hat, y
    
    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.shared_step(batch,prefix="train")
        
        return {"loss": loss, "pred_dose": y_hat, "gt_dose": y}
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.shared_step(batch,prefix="val")
        return {"loss": loss, "pred_dose": y_hat, "gt_dose": y}

    def logging_step(self,res_dict,prefix):
        for k,v in res_dict.items():
            self.log(f"{prefix}/{k}",v,prog_bar=True,sync_dist=True)



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer


    def generate_prior(self,ct, phys_source, phys_target, affine_trans,energy):
        """Prior field (B, 1, X, Y, Z) on the voxel grid of batch['ct'].
 
        self.generate_prior(batch=batch)                   geometric prior (ADoTA beam shape)
        self.generate_prior(batch=batch, kind="energy")    energy-deposition prior
        Any other keyword argument overrides the matching default in cfg below.
        """
        cfg = {"kind": "geometric",
               "sigma": 2.0,                                           # geometric: tube width [mm]
               "range_fits": {"TH": (36.99015255, 0.5, -208.1400),     # energy: L(E) = a * E**b + c [mm]
                              "AB": (0.09815625, 1.485708, 13.5085)},
               "sigma_lat0": 2.0, "sigma_lat_slope": 0.004,            # energy: lateral sigma(d) = sigma_lat0 + slope * d
               "plateau": 0.7, "sigma_entrance": 16.0, "sigma_distal": 8.0}   # energy: Bragg-like depth profile
        
        #batch = cfg["batch"]
        group = None
        X, Y, Z = ct.shape[-3:]
        kw = dict(dtype=ct.dtype)
 
        # 1. Ray end points LPS -> RAS, unit beam direction u
        flip = torch.tensor([-1.0, -1.0, 1.0], **kw)

        source = phys_source * flip
        target = phys_target * flip
        u = target - source
        u = (u / u.norm(dim=1, keepdim=True))[:, :, None, None, None]
 
        # 2. Voxel -> world map of the patch, x = M (i, j, k) + t: centred x-y crop, z slab from z_start
        A = affine_trans

        M = A[:, :3, :3, None, None, None]
        t = A[:, :3, 3] + (A[:, :3, :3] ).sum(-1)
 
        # 3. Every voxel relative to the beam: delta = depth past the target, r2 = squared distance from the axis
        i = torch.arange(X).view(1, 1, X, 1, 1)
        j = torch.arange(Y).view(1, 1, 1, Y, 1)
        k = torch.arange(Z).view(1, 1, 1, 1, Z)
        v = M[:, :, 0] * i + M[:, :, 1] * j + M[:, :, 2] * k + (t - target)[:, :, None, None, None]
        delta = (v * u).sum(1)
        r2 = (v - delta[:, None] * u).pow(2).sum(1)
 
        if cfg["kind"] == "geometric":
            field = torch.exp(-r2 / (2 * cfg["sigma"] ** 2))
        
        elif cfg["kind"] == "energy":
            # Bragg-like curve along the beam: plateau before the target, peak at it, fall-off after it
            E = energy
            a, b, c = torch.tensor([cfg["range_fits"][g] for g in group]).T
            L = (a * E**b + c).clamp(min=1.0)[:, None, None, None]    # fitted beam length
            depth = delta + L                                           # depth from the fitted source target - L u
            sigma_lat = cfg["sigma_lat0"] + cfg["sigma_lat_slope"] * depth.clamp(min=0)
            lateral = torch.exp(-r2 / (2 * sigma_lat**2))
            sigma = torch.where(delta < 0, cfg["sigma_entrance"], cfg["sigma_distal"])
            peak = torch.exp(-delta**2 / (2 * sigma**2))
            longitudinal = torch.where(delta < 0, cfg["plateau"] + (1 - cfg["plateau"]) * peak, peak) * (depth >= 0)
            field = E[:, None, None, None] * longitudinal * lateral
        return field.unsqueeze(1)
 
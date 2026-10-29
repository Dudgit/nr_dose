
#* Default system configs
import os
import argparse

import torch
from omegaconf import OmegaConf

#* Linghting and Lighning functions
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import LightningEnvironment

#* Own scripts
from scripts.data_loader_adota import create_dataloader
from scripts.callbacks import Matshow3DVisualizerCallback
from scripts.metrics import Level1LossFunction
from scripts.training_boys import DoseTrainer

torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("--config", type=str, default="default", help="Which config file to use")
parser.add_argument('--hw', type=str, default = "atlasz")
args =  parser.parse_args()

def create_config():
    cfg = OmegaConf.load(f"configs/flare_config.yaml")
    cfg = cfg[args.hw]
    if args.config != "default":
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f"configs/{args.config}_config.yaml"))
    print("Run name:", cfg['run_name'])
    return cfg


#! Here we need to load in the MODEL ÁRON!!!
def create_dota_instance(cfg):
    from src.adota.models import DoTA3D_v3

    model = DoTA3D_v3(**cfg['dotakwgs'])
    return model


def create_trainer(cfg):
    model = create_dota_instance(cfg)
    instance_model = DoseTrainer(model)
    instance_model.init_loss(cfg['lr'], Level1LossFunction(**cfg['losskwgs'])) #! Here we need to update this with the losses HILAL!
    return instance_model

def create_callbacks(cfg):
    last_callback = ModelCheckpoint(dirpath=os.path.join("checkpoints", cfg['run_name']),filename='last',save_last=True)
    matshow_callback = Matshow3DVisualizerCallback(num_samples=1)
    #! Here we can add the aditional metrics logging callbacks HILAL!
    return [last_callback, matshow_callback]

def training_start(cfg,run_name,model,train_loader,val_loader):
    callbacks = create_callbacks(cfg)
    wandb_name = run_name
    wandb_logger = WandbLogger(log_model=False, project="DoseRad", name=wandb_name,entity="ELTE_dl_competition_team",save_dir="/tmp",config=dict(cfg))
    strategy = "ddp"
    trainer = pl.Trainer(max_epochs=cfg['train']['num_epochs'],precision="bf16-mixed",logger=wandb_logger, strategy = strategy,
                                 accelerator="gpu",devices = 'auto',callbacks=callbacks,plugins=LightningEnvironment(),num_sanity_val_steps=0)
    last_ckpt_path = os.path.join("checkpoints", run_name, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"Resuming 8-hour chain from {last_ckpt_path}...")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt_path)
    else:
        print("Starting fresh training run...")
        trainer.fit(model, train_loader, val_loader)  

def train_dota():
    cfg = create_config()
    run_name_base = cfg['run_name']
    train_loader, val_loader = create_dataloader(cfg=cfg)
    model = create_trainer(cfg)
    training_start(cfg,run_name_base,model,train_loader,val_loader)

if __name__ == "__main__":
    train_dota()






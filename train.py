import argparse
from omegaconf import OmegaConf
import pytrch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import LightningEnvironment

import os


from scripts.data_loader import get_loaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, default="default", help="Which config file to use")
    return parser.parse_args()


def create_dota_instance(cfg):
    from scripts.models import DoTA_based
    from scripts.train_backbone import DoseTrainer
    model = DoTA_based(**cfg['modelkwgs'])
    dose_instance_model = DoseTrainer(model)
    return dose_instance_model

def train_dota():
    run_name ="Vanilla"
    cfg = OmegaConf.load(f"configs/default_config.yaml")
    
    train_loader, val_loader = get_loaders()
    model = create_dota_instance(cfg)
    
    last_callback = ModelCheckpoint(dirpath=os.path.join("checkpoints", run_name),filename='last',save_last=True)
    wandb_logger = WandbLogger(log_model=True, project="DOTA", name=run_name,entity="DOSERAD",save_dir="/tmp")
    
    trainer = pl.Trainer(max_epochs=cfg['train']['num_epochs'],precision="bf16-mixed",logger=wandb_logger,accelerator="gpu",callbacks=[last_callback],plugins=LightningEnvironment())
    last_ckpt_path = os.path.join("checkpoints", run_name, "last.ckpt")
    
    if os.path.exists(last_ckpt_path):
        print(f"Resuming 8-hour chain from {last_ckpt_path}...")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt_path)
    else:
        print("Starting fresh training run...")
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train_dota()

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
from scripts.data_loader import get_loaders
from scripts.callbacks import DoseLevel1MetricsCallback, Matshow3DVisualizerCallback
from scripts.metrics import Level1LossFunction
from scripts.train_backbone import DoseTrainer

torch.set_float32_matmul_precision('medium')


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, default="default", help="Which config file to use")
    return parser.parse_args()



def create_config():
    cfg = OmegaConf.load(f"configs/default_config.yaml")
    args = parse_args()
    if args.config != "default":
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f"configs/{args.config}.yaml"))
    return cfg

def create_dota_instance(cfg):
    from scripts.models import DoTA_based
    model = DoTA_based(**cfg['dotakwgs'])
    return model

def create_unet_instance(cfg):
    print('creating unet')
    from scripts.models import ConditionalDoseUNet
    model = ConditionalDoseUNet(**cfg['unetkwgs'])
    return model

def create_filmUnet(cfg):
    from scripts.models import FiLMConditionalDoseUnet
    model = FiLMConditionalDoseUnet(**cfg['filmkwgs'])
    return model 

def create_attentionUnet_instance(cfg):
    from scripts.models import DoseAttentionUnet
    model = DoseAttentionUnet(**cfg['attentionkwgs'])
    return model

def choseModels(cfg):
    if cfg['modelname'] == "DoTA":
        model = create_dota_instance(cfg)
    elif cfg['modelname'] == "ConditionalDoseUNet":
        model = create_unet_instance(cfg)
    elif cfg['modelname'] == "FiLMConditionalDoseUnet":
        model = create_filmUnet(cfg)
    elif cfg['modelname'] == "DoseAttentionUnet":
        model = create_attentionUnet_instance(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg['modelname']}")
    loss_function = Level1LossFunction(**cfg['losskwgs'])
    dose_instance_model = DoseTrainer(model, loss_function)
    return dose_instance_model

def create_callbacks(cfg):
    last_callback = ModelCheckpoint(dirpath=os.path.join("checkpoints", cfg['run_name']),filename='last',save_last=True)
    doe_level1_callback = DoseLevel1MetricsCallback()
    matshow_callback = Matshow3DVisualizerCallback(num_samples=1)
    return [last_callback, doe_level1_callback, matshow_callback]

def train_dota():
    cfg = create_config()
    run_name =cfg['run_name']

    train_loader, val_loader = get_loaders()
    model = choseModels(cfg)
    callbacks = create_callbacks(cfg)

    wandb_logger = WandbLogger(log_model=True, project="DoseRad", name=run_name,entity="ELTE_dl_competition_team",save_dir="/tmp")
    
    trainer = pl.Trainer(max_epochs=cfg['train']['num_epochs'],precision="bf16-mixed",logger=wandb_logger,
                         accelerator="gpu",devices=4,callbacks=callbacks,plugins=LightningEnvironment(),num_sanity_val_steps=0)
    
    last_ckpt_path = os.path.join("checkpoints", run_name, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"Resuming 8-hour chain from {last_ckpt_path}...")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt_path)
    else:
        print("Starting fresh training run...")
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    train_dota()
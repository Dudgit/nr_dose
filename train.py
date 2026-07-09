
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
from scripts.train_backbone import DoseTrainer, DoseGANTrainer

torch.set_float32_matmul_precision('medium')


parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("--config", type=str, default="default", help="Which config file to use")
parser.add_argument('--hw', type=str, default = "atlasz")
args =  parser.parse_args()



def create_config():
    cfg = OmegaConf.load(f"configs/hw_config.yaml")
    cfg = cfg[args.hw]
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
    if cfg['train']['adversarial']['use']:
        from monai.networks.nets import PatchDiscriminator
        discriminator = PatchDiscriminator(
        spatial_dims=3,in_channels=3, # 1 for CT + 1 for Dose (It needs to see the anatomy AND the dose to judge reality)
        num_layers_d=3,channels=64,norm="instance")
        dose_instance_model =DoseGANTrainer(generator=model,discriminator=discriminator,loss_function=loss_function,
                                            adv_weight=cfg['train']['adversarial']['adv_weight'],d_update_freq=cfg['train']['adversarial']['d_update_freq'],lr=cfg['train']['lr'],
                                            start_epoch=cfg['train']['adversarial']['start_epoch'],ramp_length=cfg['train']['adversarial']['ramp_length'],
                                            max_mae_weight=cfg['train']['adversarial']['max_mae_weight'],def_mae_weight=cfg['train']['adversarial']['def_mae_weight'])
    else:
        dose_instance_model = DoseTrainer(model, loss_function,lr = cfg['train']['lr'],use_warmups=cfg['train']['use_warmups'])

    return dose_instance_model

def create_callbacks(cfg):
    last_callback = ModelCheckpoint(dirpath=os.path.join("checkpoints", cfg['run_name']),filename='last',save_last=True)
    doe_level1_callback = DoseLevel1MetricsCallback()
    matshow_callback = Matshow3DVisualizerCallback(num_samples=1)
    return [last_callback, doe_level1_callback, matshow_callback]

def train_dota():
    cfg = create_config()
    run_name_base = cfg['run_name'] #if args.hw == "atlasz" else cfg['komondor_run_name']
    run_name_base = run_name_base + "_adv" if cfg['train']['adversarial']['use'] else run_name_base
    modelname = cfg['modelname']# if args.hw == "atlasz" else cfg['komondor_modelname']
    run_name = args.hw + "_" + run_name_base + "_" + str(modelname) + "_" + str(cfg['train']['num_epochs']) + "epochs"
    cfg['run_name'] = run_name
    train_loader, val_loader = get_loaders(hw = args.hw)
    model = choseModels(cfg)
    callbacks = create_callbacks(cfg)

    wandb_logger = WandbLogger(log_model=True, project="DoseRad", name=run_name,entity="ELTE_dl_competition_team",save_dir="/tmp",config=dict(cfg))
    strategy = "ddp" if not cfg['train']['adversarial']['use'] else "ddp_find_unused_parameters_true" 
    fine_tune_steps = 50 if cfg['train']["fine_tune"] else 0
    trainer = pl.Trainer(max_epochs=cfg['train']['num_epochs']+fine_tune_steps,precision="bf16-mixed",logger=wandb_logger, strategy = strategy,
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
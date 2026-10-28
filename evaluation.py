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
from scripts.callbacks import DoseLevel1MetricsCallback, Matshow3DVisualizerCallback, BraggPeakDistanceCallback
from scripts.metrics import Level1LossFunction
from scripts.train_backbone import DoseTrainer, DoseGANTrainer


torch.set_float32_matmul_precision('medium')


parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("--config", type=str, default="default", help="Which config file to use")
parser.add_argument('--hw', type=str, default = "atlasz")
args =  parser.parse_args()



def create_config():
    cfg = OmegaConf.load(f"configs/default_config.yaml")
    cfg = cfg[args.hw]
    if args.config != "default":
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f"configs/{args.config}_config.yaml"))
    print("Run name:", cfg['run_name'])
    return cfg

def create_dota_instance(cfg):
    from scripts.models import DoTA_based
    model = DoTA_based(**cfg['dotakwgs'])
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
        dose_instance_model =DoseGANTrainer(generator=model,discriminator=discriminator,loss_function=loss_function,lr=cfg['train']['lr'],
                                            **cfg['train']['adversarial'])
    else:
        dose_instance_model = DoseTrainer(model, loss_function,lr = cfg['train']['lr'],use_warmups=cfg['train']['use_warmups'])

    return dose_instance_model




model = choseModels(create_config())


##validation loader
##train_loader, val_loader = get_loaders(create_config())

##végigmegyünk a validation loaderen, legeneráljuk a predikciókat, és elmentjük őket egy fájlba, hogy később kiértékelhessük őket 
##(mindig meg kell nézni hogy mi az input shape, scaling, normálás, stb, predicted doseokat elég csak felszorozni 10 000-el)


##data_loader = get_loaders(create_config(),train=False)
##reshape adott sample id-ben benne van hogy mi a spacingje, ezt kiolvasom, abba reshapelem 
##amikor az xct-n, zct-n megyek, akkor a megfelelő spacing kell


import json
##beolvasom
def generate_results(model):
    cfg = create_config()
    cfg["batch_size"] = 8

    shape_dict = json.load(open("shapes.json"), "r")
    _, val_loader = get_loaders(cfg, hw = "komondor")

    for batch in val_loader:
        model.predict(batch)


##prediktálok CT, priorokból, dózist (input: CT, priorok) output: dózis
##az x, z, y, spacing, scaling, stb. mindent kiolvasok
##reshapelem ennek megfelelően
##az értékekeket meg visszaszorzom a skálázó értékkel, hogy visszakapjam a valós dózist
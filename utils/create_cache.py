from scripts.data_loader import get_loaders
import tqdm
from omegaconf import OmegaConf

if __name__ == "__main__":
    cfg = OmegaConf.load(f"configs/default_config.yaml")
    cfg = cfg['komondor']
    train_loader, val_loader = get_loaders(cfg = cfg, hw = "komondor")
    for batch in tqdm.tqdm(train_loader):
        continue
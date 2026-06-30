from scripts.data_loader import get_loaders
import tqdm

if __name__ == "__main__":
    train_loader, val_loader = get_loaders(hw = "atlasz")
    for batch in tqdm.tqdm(train_loader):
        continue
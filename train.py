import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, default="default", help="Which config file to use")
    return parser.parse_args()
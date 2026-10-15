import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

# waiting on the data and model files to be added back in

#from data import get_loaders, LABEL_NAMES
#from model import FireSmokeResNet

LABEL_NAMES = ["fire", "smoke", "fire_and_smoke", "neither"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def run_checks():
    print("no checks yet")
    

def main():
    args = parse_args()
    if args.check:
        run_checks()
        return
    print(args)
    
    
if __name__ == "__main__":
    main()
    
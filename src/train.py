import argparse
import csv
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader   # batching/shuffling only - network itself is hand-built

from model.resnet import ResNet            # ResNet class built layer by layer in src/model/resnet.py
from data import get_loaders, LABEL_NAMES  # real fire/smoke data + class names (index = label 0-3)


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
    p.add_argument("--no-class-weights", action="store_true")
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


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU if available


def make_dummy_loaders(batch_size, img_size=64, n=256):
    # fake data in the same format as get_loaders() - for fast crash-testing only
    def make_loader(shuffle):
        x = torch.randn(n, 3, img_size, img_size)          # n fake RGB images (batch, channels, H, W)
        y = torch.randint(0, len(LABEL_NAMES), (n,))        # n random class indices, 0-3
        dataset = TensorDataset(x, y)                       # pair each image with its label
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return make_loader(True), make_loader(False), make_loader(False)


def class_counts(loader):
    # count how many images of each class a loader holds (works with Subset or full dataset)
    ds = loader.dataset
    if hasattr(ds, "indices"):                               # Subset: look up only its indices
        labels = [ds.dataset.samples[i][1] for i in ds.indices]
    else:
        labels = [label for _, label in ds.samples]
    counts = Counter(labels)
    return [counts.get(c, 0) for c in range(len(LABEL_NAMES))]


def class_weights(counts):
    # inverse-frequency weights: total / (num_classes * count) -> rare classes weigh more
    total, k = sum(counts), len(counts)
    return torch.tensor([total / (k * c) for c in counts], dtype=torch.float32)


def build_optimizer(model, args):
    if args.optimizer == "adamw":
        lr = args.lr if args.lr is not None else 1e-3
        wd = args.weight_decay if args.weight_decay is not None else 0.01
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        lr = args.lr if args.lr is not None else 0.1
        wd = args.weight_decay if args.weight_decay is not None else 5e-4
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    return opt, lr, wd


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                                   # training mode: BatchNorm uses batch stats, Dropout active
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)   # move batch to GPU/CPU

        optimizer.zero_grad()                       # clear gradients from the previous batch
        logits = model(images)                      # forward pass: raw scores, shape (batch, 4)
        loss = criterion(logits, labels)            # weighted cross-entropy vs true labels
        loss.backward()                             # backprop: gradient of loss w.r.t. every weight
        optimizer.step()                            # update weights using those gradients

        total_loss += loss.item() * labels.size(0)  # x batch size so the average is per image
        correct += (logits.argmax(dim=1) == labels).sum().item()   # predicted class = highest score
        seen += labels.size(0)
    return total_loss / seen, correct / seen        # average train loss, train accuracy


def validate(model, loader, criterion, device):
    model.eval()                                    # eval mode: BatchNorm uses running stats, Dropout off
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():                           # no gradients needed -> faster, less memory
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen        # average val loss, val accuracy


def run_checks():
    model = ResNet(num_classes=len(LABEL_NAMES))      # one output per class
    x = torch.randn(2, 3, 64, 64)                      # fake batch: 2 RGB 64x64 images
    out = model(x)                                     # forward pass
    print("Output shape:", out.shape)                  # expect torch.Size([2, 4])


def main():
    args = parse_args()
    if args.check:
        run_checks()
        return

    set_seed(args.seed)                                      # reproducible runs
    device = get_device()
    print("Device:", device)

    # ---- data ----
    if args.dummy:
        train_dl, val_dl, test_dl = make_dummy_loaders(args.batch_size)
        counts = None
    else:
        train_dl, val_dl, test_dl = get_loaders(args.batch_size, not args.no_augment,
                                                args.num_workers, args.seed)
        counts = class_counts(train_dl)
        print("Train counts:", dict(zip(LABEL_NAMES, counts)))

    # ---- model ----
    model = ResNet(num_classes=len(LABEL_NAMES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ---- loss ----
    if counts is not None and not args.no_class_weights:
        weights = class_weights(counts).to(device)
        print("Class weights:", dict(zip(LABEL_NAMES, [round(w, 2) for w in weights.tolist()])))
        criterion = nn.CrossEntropyLoss(weight=weights)     # weighted: rare classes cost more
    else:
        criterion = nn.CrossEntropyLoss()                    # unweighted

    # ---- optimiser ----
    optimizer, lr, wd = build_optimizer(model, args)
    print(f"Optimizer: {args.optimizer}  lr={lr}  weight_decay={wd}")

    # ---- learning-rate schedule ----
    # cosine annealing: lr starts at its initial value and smoothly decays towards 0 by the last epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- output files ----
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)                 # create results/ if it doesn't exist
    csv_path = os.path.join(results_dir, f"{args.run_name}.csv")
    ckpt_path = os.path.join(results_dir, f"{args.run_name}_best.pt")
    with open(csv_path, "w", newline="") as f:               # new file each run, header row first
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"])

    # ---- training loop ----
    best_val_loss = math.inf                                 # best (lowest) val loss seen so far
    epochs_without_improvement = 0                           # counter for early stopping
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]             # lr used during this epoch
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_dl, criterion, device)
        scheduler.step()                                     # decay lr for the next epoch
        elapsed = time.time() - start

        # log this epoch to the CSV (used later for loss/accuracy plots)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                                    f"{va_loss:.4f}", f"{va_acc:.4f}", f"{lr_now:.6f}", f"{elapsed:.0f}"])

        # checkpoint: keep only the model with the lowest validation loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = "  * saved best"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | lr {lr_now:.5f} | {elapsed:.0f}s{marker}")

        # early stopping: stop if val loss hasn't improved for `patience` epochs in a row
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: no val improvement for {args.patience} epochs.")
            break

    print(f"Best val loss {best_val_loss:.4f}")
    print(f"Log:        {os.path.normpath(csv_path)}")
    print(f"Best model: {os.path.normpath(ckpt_path)}")


if __name__ == "__main__":
    main()

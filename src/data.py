import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

LABEL_NAMES = ["fire", "smoke", "fire_and_smoke", "neither"]
DATA_ROOT = r"C:\Users\louis\Desktop\NeuralNetworks\Project"
IMG_SIZE = 64
SMOKE_ID, FIRE_ID = 0, 1          # D-Fire YOLO class ids - verify visually


def label_from_file(path):
    with open(path) as f:
        ids = {int(line.split()[0]) for line in f if line.strip()}   # set of box classes in this image
    has_fire, has_smoke = FIRE_ID in ids, SMOKE_ID in ids
    if has_fire and has_smoke: return 2
    if has_fire:               return 0
    if has_smoke:              return 1
    return 3                                                        # empty file = neither


class FireSmokeDataset(Dataset):
    def __init__(self, split, transform):
        img_dir = os.path.join(DATA_ROOT, split, "images")
        lbl_dir = os.path.join(DATA_ROOT, split, "labels")
        self.samples = []
        for name in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(name)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            label = label_from_file(lbl_path) if os.path.exists(lbl_path) else 3
            self.samples.append((os.path.join(img_dir, name), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")      # some images may be greyscale/RGBA
        return self.transform(img), label


def get_loaders(batch_size, augment=True, num_workers=4, seed=42, val_frac=0.15):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(), norm,
    ]) if augment else eval_tf

    # two views of the same train folder: augmented for training, clean for validation
    train_full = FireSmokeDataset("train", train_tf)
    val_full = FireSmokeDataset("train", eval_tf)
    idx = list(range(len(train_full)))
    random.Random(seed).shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_set, train_set = Subset(val_full, idx[:n_val]), Subset(train_full, idx[n_val:])
    test_set = FireSmokeDataset("test", eval_tf)

    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers)
    return mk(train_set, True), mk(val_set, False), mk(test_set, False)
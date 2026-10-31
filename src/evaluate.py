import os
import torch
from model.resnet import ResNet
from data import get_loaders, LABEL_NAMES

RUNS = ["baseline", "sgd", "no_aug"]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = len(LABEL_NAMES)

_, _, test_dl = get_loaders(batch_size=64, augment=False, num_workers=0)

summary = []

for run in RUNS:
    ckpt = os.path.join(RESULTS, f"{run}_best.pt")
    if not os.path.exists(ckpt):
        print(f"\n== {run}: missing {ckpt}, skipping")
        continue

    model = ResNet(num_classes=n).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    confusion = torch.zeros(n, n, dtype=torch.long)   # rows = true, cols = predicted
    with torch.no_grad():
        for x, y in test_dl:
            pred = model(x.to(device)).argmax(1).cpu()
            confusion += torch.bincount(y * n + pred, minlength=n * n).view(n, n)

    acc = confusion.diag().sum().item() / confusion.sum().item()
    print(f"\n== {run}: test accuracy {acc:.4f}")

    precisions, recalls, f1s = [], [], []
    for i, name in enumerate(LABEL_NAMES):
        tp = confusion[i, i].item()
        total = confusion[i].sum().item()          # true count for this class
        predicted = confusion[:, i].sum().item()   # predicted count for this class
        recall = tp / total if total else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f"  {name:15s} P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}  ({total} images)")

    macro_p, macro_r, macro_f1 = (sum(v) / n for v in (precisions, recalls, f1s))
    print(f"  macro P {macro_p:.4f}  macro R {macro_r:.4f}  macro F1 {macro_f1:.4f}")
    print(confusion)
    summary.append((run, acc, macro_p, macro_r, macro_f1))

print("\n== Summary")
print(f"{'run':10s} {'acc':>7s} {'macroP':>8s} {'macroR':>8s} {'macroF1':>8s}")
for run, acc, p, r, f in summary:
    print(f"{run:10s} {acc:7.4f} {p:8.4f} {r:8.4f} {f:8.4f}")
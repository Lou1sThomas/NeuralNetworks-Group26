"""Turn the training-log CSVs in results/ into report figures and a summary table.

Usage (from fire-smoke-cnn/src, venv active):
    python plot_results.py

Reads results/<run>.csv for each run in RUNS and saves PNGs to results/figures/.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")                       # save to files, no window needed
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# run name -> label used in legends
RUNS = {
    "baseline": "Baseline (AdamW, aug, class weights)",
    "sgd": "SGD + momentum",
    "no_aug": "No augmentation",
    "no_weights": "No class weights",
}
COLOURS = {"baseline": "#1f77b4", "sgd": "#d62728", "no_aug": "#2ca02c", "no_weights": "#9467bd"}


def load(run):
    """Read one run's CSV into a dict of lists (epoch, train_loss, ...). Returns None if missing."""
    path = os.path.join(RESULTS_DIR, f"{run}.csv")
    if not os.path.exists(path):
        print(f"  (skipping {run}: {path} not found)")
        return None
    cols = {}
    # utf-8-sig strips a hidden BOM that Excel adds when it saves a CSV
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","   # Excel in some locales saves with ;
        for row in csv.DictReader(f, delimiter=delim):
            for k, v in row.items():
                if k is None or v is None or not v.strip():
                    continue
                k = k.strip().lower()                       # tolerate " val_loss" / "Val_Loss"
                cols.setdefault(k, []).append(int(float(v)) if k == "epoch" else float(v))
    needed = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    missing = [c for c in needed if c not in cols]
    if missing:
        print(f"  (skipping {run}: columns {missing} missing; found {list(cols)})")
        return None
    return cols


def best_epoch(d):
    """Epoch with the lowest val loss = the epoch saved as <run>_best.pt."""
    i = min(range(len(d["val_loss"])), key=lambda j: d["val_loss"][j])
    return i


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {os.path.normpath(path)}")


def plot_single_run(d, run, name):
    """Train vs val loss and accuracy for one run, with the best epoch marked."""
    b = best_epoch(d)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(ax1, "loss", "Loss"), (ax2, "acc", "Accuracy")]:
        ax.plot(d["epoch"], d[f"train_{metric}"], "-o", ms=3, label="Train")
        ax.plot(d["epoch"], d[f"val_{metric}"], "-s", ms=3, label="Validation")
        ax.axvline(d["epoch"][b], color="grey", ls="--", lw=1, label=f"Best epoch ({d['epoch'][b]})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{RUNS[run]}: {title.lower()}")
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, name)


def plot_compare(data, runs, metrics, name, title):
    """Compare runs on the same axes. metrics: list of (column, ylabel) - one subplot each."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4))
    axes = axes if len(metrics) > 1 else [axes]
    for ax, (col, ylabel) in zip(axes, metrics):
        for run in runs:
            if data.get(run) is None or col not in data[run]:
                continue
            d = data[run]
            ax.plot(d["epoch"], d[col], "-o", ms=3, color=COLOURS[run], label=RUNS[run])
            b = best_epoch(d)
            ax.axvline(d["epoch"][b], color=COLOURS[run], ls=":", lw=1.2)   # best epoch (lowest val loss)
        if col == "lr":
            ax.set_yscale("log")        # SGD (0.1) and AdamW (0.001) differ 100x - log scale shows both
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title + "\n(dotted line = best epoch, lowest val loss)", fontsize=11)   # 2 lines so it fits narrow figures
    save(fig, name)


def plot_train_val_gap(data, runs, name, title):
    """Train (dashed) and val (solid) loss for several runs - shows when each starts to overfit."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        if data.get(run) is None:
            continue
        d = data[run]
        ax.plot(d["epoch"], d["train_loss"], "--", color=COLOURS[run], label=f"{RUNS[run]} - train")
        ax.plot(d["epoch"], d["val_loss"], "-o", ms=3, color=COLOURS[run], label=f"{RUNS[run]} - val")
        b = best_epoch(d)
        ax.axvline(d["epoch"][b], color=COLOURS[run], ls=":", lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title + "\n(dotted line = best epoch / start of overfitting)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, name)


def summary_table(data):
    """Print the key numbers for the report tables."""
    print("\n{:<12} {:>6} {:>10} {:>12} {:>14} {:>12} {:>12}".format(
        "run", "best", "stopped", "best val", "val acc", "final train", "final val"))
    print("{:<12} {:>6} {:>10} {:>12} {:>14} {:>12} {:>12}".format(
        "", "epoch", "at", "loss", "at best", "acc", "acc"))
    print("-" * 84)
    for run, d in data.items():
        if d is None:
            continue
        b = best_epoch(d)
        print("{:<12} {:>6} {:>10} {:>12.4f} {:>13.1f}% {:>11.1f}% {:>11.1f}%".format(
            run, d["epoch"][b], d["epoch"][-1], d["val_loss"][b], 100 * d["val_acc"][b],
            100 * d["train_acc"][-1], 100 * d["val_acc"][-1]))
    print("\nNote: loss values are not comparable between weighted (baseline, sgd, no_aug)"
          "\nand unweighted (no_weights) cross-entropy - compare accuracy / test metrics instead.")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Loading CSVs from", os.path.normpath(RESULTS_DIR))
    data = {run: load(run) for run in RUNS}

    if data["baseline"] is not None:
        # Section 4: the final model's learning curves
        plot_single_run(data["baseline"], "baseline", "fig_baseline_curves.png")

    # Section 3.b: optimiser comparison
    plot_compare(data, ["baseline", "sgd"],
                 [("val_loss", "Validation loss"), ("val_acc", "Validation accuracy")],
                 "fig_optimiser_adamw_vs_sgd.png", "AdamW vs SGD with momentum")
    if data["sgd"] is not None and "lr" in data["sgd"]:
        plot_compare(data, ["baseline", "sgd"], [("lr", "Learning rate")],
                     "fig_lr_schedules.png", "Cosine learning-rate schedules")
    else:
        print("  (not making fig_lr_schedules.png: sgd.csv missing or has no 'lr' column)")

    # Section 5 (a): augmentation - train vs val loss shows the earlier overfitting
    plot_train_val_gap(data, ["baseline", "no_aug"], "fig_aug_vs_noaug_loss.png",
                       "Effect of data augmentation on overfitting")
    plot_compare(data, ["baseline", "no_aug"],
                 [("train_acc", "Train accuracy"), ("val_acc", "Validation accuracy")],
                 "fig_aug_vs_noaug_acc.png", "Effect of data augmentation on accuracy")

    # Section 5 (c): class weights - accuracy only (losses are on different scales)
    plot_compare(data, ["baseline", "no_weights"],
                 [("train_acc", "Train accuracy"), ("val_acc", "Validation accuracy")],
                 "fig_weights_vs_noweights_acc.png", "Effect of class weights on accuracy")

    # Overview: validation accuracy of every run
    plot_compare(data, list(RUNS), [("val_acc", "Validation accuracy")],
                 "fig_all_runs_val_acc.png", "Validation accuracy, all runs")

    summary_table(data)


if __name__ == "__main__":
    main()
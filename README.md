# Fire & Smoke Detection CNN — CS4287 Assignment 1

Team: _Name 1 (ID)_, _Name 2 (ID)_, _Name 3 (ID)_
Deadline: **23:59 Saturday 31 October 2026** (Brightspace)

A CNN that classifies an image as **fire**, **smoke**, **both** or **neither**.
Dataset: D-Fire style YOLO-format data (`train/` ~17k images, `test/` ~4.3k images).
Label files: one line per box, `class x_center y_center width height`; class `0` = smoke, `1` = fire; an empty file = no fire/smoke.

## Repo layout
```
notebooks/   Colab notebooks (00_setup_colab.ipynb = shared setup; final notebook goes here)
src/         Reusable Python helpers (data loading, model builders, plotting)
results/     Small outputs worth keeping: plots, metric CSVs (no model weights)
report/      Report drafts / figures for the PDF
docs/        genai_log.md (required by the brief) and team notes
```
The dataset and model weights are **not** in Git — they live in the shared Google Drive folder.

## One-time setup (each team member)
1. Accept the GitHub collaborator invite.
2. In Google Drive, open the shared folder **CS4287-Project**, right-click → *Organise → Add shortcut* → *My Drive*.
   (This makes the path `/content/drive/MyDrive/CS4287-Project/dfire.zip` work for everyone.)
3. Go to colab.research.google.com → *GitHub* tab → tick *Include private repos* → open `notebooks/00_setup_colab.ipynb`.
4. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.

## Working rules (to avoid merge conflicts in .ipynb files)
- Notebooks are JSON, so two people editing the same notebook = painful conflicts.
  **Each person works in their own notebook** (e.g. `notebooks/louis_experiments.ipynb`).
  Shared code goes in `src/*.py`, which merges cleanly.
- One person owns the final submission notebook at a time — say so in the group chat before editing it.
- Save from Colab with *File → Save a copy in GitHub*, write a clear commit message.
- Before committing a notebook: *Edit → Clear all outputs* (keeps diffs small) unless you need the plots in it.
- Pull/refresh before you start working; never commit data or `.h5`/`.keras` files.
- Log every Generative AI prompt in `docs/genai_log.md` as you go (the brief requires ALL prompts).

## Final submission checklist (from the brief)
- Notebook named `CS4287-Assign2-ID1-ID2-ID3.ipynb` (brief's wording — check with lecturer whether it should be Assign1)
- Line 1: comment with names + IDs · Line 2: comment saying whether it runs to the end · Line 3: link to any reused third-party source
- Every critical line commented by us; PDF report per the brief's section list

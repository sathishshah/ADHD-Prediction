# How to Run the ADHD EEG Pipeline — Step-by-Step

---

## Step 1 — Download the Dataset

You need **121 `.mat` files** (61 ADHD + 60 Control children).

### Option A — Kaggle (Recommended)

1. Go to [https://www.kaggle.com](https://www.kaggle.com) and create a free account if you don't have one.
2. In the search bar, type: `EEG data ADHD control children`
3. Look for a dataset that mentions **121 files**, **19 channels**, **128 Hz**.
4. Click **Download** — you will get a `.zip` file.
5. Extract the zip. You should see folders like `ADHD/`, `Control/`, or `ADHD_part1/`, etc.

### Option B — IEEE Dataport

1. Go to: `https://ieee-dataport.org/open-access/eeg-data-adhd-control-children`
2. Create a free IEEE account.
3. Click **Download** on the dataset page.
4. Extract the downloaded archive.

### What You Should Have After Extraction

- A folder with **61 ADHD `.mat` files**
- A folder with **60 Control `.mat` files**
- Total = **121 files**

> If you see a very different count (e.g., 10 files, 200 files), you have the wrong dataset — try another mirror.

---

## Step 2 — Place Files into the Project

Inside `C:\dev\ADHD Prediction\`, there is a `data\` folder already created.

Copy your downloaded files into:

```
C:\dev\ADHD Prediction\
└── data\
    ├── ADHD\
    │   ├── v1p.mat
    │   ├── v2p.mat
    │   └── ... (61 files total)
    └── Control\
        ├── v1c.mat
        ├── v2c.mat
        └── ... (60 files total)
```

**How to do it:**
1. Open File Explorer.
2. Navigate to your extracted dataset folder.
3. Copy all ADHD `.mat` files → paste into `C:\dev\ADHD Prediction\data\ADHD\`
4. Copy all Control `.mat` files → paste into `C:\dev\ADHD Prediction\data\Control\`

> The pipeline will auto-detect messy mirror layouts (ADHD_part1, ADHD_part2, etc.)
> but organizing them manually is safest.

---

## Step 3 — Open a Terminal

1. Press `Win + R`, type `cmd` or `powershell`, press Enter.
2. Navigate to the project folder:

```powershell
cd "C:\dev\ADHD Prediction"
```

---

## Step 4 — Install Dependencies

Run this command once:

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow (deep learning)
- scikit-learn (classical ML)
- SHAP (explainability)
- MNE, SciPy, NumPy, pandas, matplotlib, tqdm

> **Requires Python 3.10 or 3.11.** Check yours with: `python --version`

---

## Step 5 — Run the Smoke Test First

Before doing the full run, always test the pipeline end-to-end with reduced settings:

```bash
python -m adhd_pipeline.run_all --quick
```

**What `--quick` does:**
- Reduces training to 10 epochs (instead of 60)
- Uses 5 folds (instead of 10)
- Limits SHAP to 50 samples
- Finishes in ~10 minutes

**What to check:**
- No error messages
- You see fold-level accuracy printed for each model
- `results/` folder is created with output files

> Results from `--quick` are for debugging only — do NOT paste them into the paper.

---

## Step 6 — Run the Full Pipeline

Once the smoke test passes, run the full experiment:

```bash
python -m adhd_pipeline.run_all --data-dir ./data --out-dir ./results
```

**Expected runtime:**

| Hardware | Time |
|---|---|
| Kaggle / Colab T4 GPU | ~90 minutes |
| Local GTX 1050 (4 GB) | ~4–6 hours |
| CPU only | ~24 hours (not recommended) |

**What it does (in order):**
1. Loads and validates 61 + 60 `.mat` files
2. Filters (bandpass + notch) and epochs each recording
3. Builds 3 cross-validation split strategies
4. Trains 3 classical models × 2 protocols
5. Trains 3 deep models × 2 protocols
6. Computes leakage gap and statistical tests
7. Runs SHAP explainability on CNN-BiLSTM
8. Writes all outputs to `results/`

---

## Step 6a — Optional: Safe Run with Checkpointing (Colab / slow machines)

If you are on Google Colab or worried about disconnection:

```bash
python -m adhd_pipeline.run_all --checkpoint
```

Saves fold results after each fold so a crash doesn't lose progress.

---

## Step 6b — Optional: Run Only One Protocol

To run only the subject-wise (leakage-free) protocol:

```bash
python -m adhd_pipeline.run_all --protocols b
```

Options: `a` (segment-wise), `b` (subject-wise), `loso` — combine with commas:

```bash
python -m adhd_pipeline.run_all --protocols a,b
```

---

## Step 7 — Check the Outputs

After the full run, verify these files exist in `results\`:

```
results\
├── all_results.json            ← all metrics (every model, every fold)
├── macro_replacements.tex      ← ready-to-paste LaTeX commands
├── folds.json                  ← fold assignments for reproducibility
├── shap_channel_importance.npy ← raw SHAP per channel
├── shap_band_importance.npy    ← raw SHAP per frequency band
└── figures\
    ├── leakage_bars.csv        ← Protocol A vs B (for Figure 4)
    ├── baseline_bars.csv       ← per-model accuracy (for Figure 5)
    ├── roc_aucs.csv            ← per-model AUC
    ├── confusion_matrix.csv    ← pooled confusion matrix
    └── shap_channels.csv       ← top channels by SHAP (for Figure 8)
```

**Acceptance checks (pipeline prints these automatically):**
- Protocol A CNN-BiLSTM accuracy should be **85–99%**
- Protocol B CNN-BiLSTM accuracy should be **65–88%**
- If either is outside range, there is likely a bug — check `RUN_NOTES.md`

---

## Step 8 — Update the Paper (`main.tex`)

This step is manual.

1. Open `main.tex` in your LaTeX editor.
2. Find the macro/placeholder block (around lines 50–95).
3. Open `results\macro_replacements.tex`.
4. **Delete** the placeholder block in `main.tex`.
5. **Paste** the entire content of `macro_replacements.tex` in its place.
6. Update the bar-chart figure coordinates:
   - Figure 4 (leakage gap) → use values from `leakage_bars.csv`
   - Figure 5 (baseline comparison) → use values from `baseline_bars.csv`
   - Figure 8 (SHAP channels) → use values from `shap_channels.csv`
7. Recompile the PDF:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

8. Read the paper end-to-end once. If real numbers differ significantly from the
   placeholder discussion text, rewrite those interpretive sentences honestly.

---

## Quick Reference — All Commands

```bash
# Navigate to project
cd "C:\dev\ADHD Prediction"

# Install packages (once)
pip install -r requirements.txt

# Smoke test (always run this first)
python -m adhd_pipeline.run_all --quick

# Full run
python -m adhd_pipeline.run_all --data-dir ./data --out-dir ./results

# Full run with checkpointing (Colab / slow machines)
python -m adhd_pipeline.run_all --checkpoint

# Partial run (subject-wise protocol only)
python -m adhd_pipeline.run_all --protocols b

# Skip SHAP (faster, for debugging models only)
python -m adhd_pipeline.run_all --no-shap
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `AssertionError: Expected 61 ADHD + 60 Control` | Wrong dataset mirror — recount your `.mat` files |
| `ModuleNotFoundError: tensorflow` | Run `pip install -r requirements.txt` again |
| Out of memory (GPU) | Add `--quick` or reduce `DL_BATCH` in `adhd_pipeline/config.py` to 16 |
| Colab disconnected mid-run | Re-run with `--checkpoint` — completed folds are skipped |
| Protocol B acc < 65% or > 88% | Likely a data-loading bug — check channel orientation in `data_io.py` |
| SHAP step crashes | Re-run with `--no-shap` to get all other results first |

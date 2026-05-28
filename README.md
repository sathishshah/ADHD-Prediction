# CNN-BiLSTM EEG-Based ADHD Prediction Pipeline

Subject-independent, leakage-aware pipeline for reproducing the results in the paper
*"EEG-Based ADHD Prediction Using CNN-BiLSTM with Leakage-Aware Evaluation"*
(Geethapriya M, Kalyanaraman P — SCOPE, VIT Vellore).

---

## Quick start

### 1. Get the dataset

Download the Nasrabadi ADHD/Control EEG dataset (121 `.mat` files, 19 channels, 128 Hz).

- **IEEE Dataport:** https://ieee-dataport.org/open-access/eeg-data-adhd-control-children  
- **Kaggle mirror:** search "EEG data for ADHD control children" — pick the mirror with 61 ADHD + 60 Control `.mat` files.

Place files so the directory looks like:

```
data/
├── ADHD/      ← 61 .mat files
└── Control/   ← 60 .mat files
```

The pipeline will auto-consolidate multi-part mirror layouts (ADHD_part1, ADHD_part2, …).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10 or 3.11 required (TensorFlow 2.15+ constraint).

### 3. Smoke test (recommended first)

```bash
python -m adhd_pipeline.run_all --quick
```

Finishes in ~10 minutes on a T4 GPU. Results from `--quick` are **not** suitable for the paper.

### 4. Full run

```bash
python -m adhd_pipeline.run_all --data-dir ./data --out-dir ./results
```

Expected time: ~90 min on Kaggle T4 / ~4–6 h on GTX 1050.

### 5. Colab / disconnection-safe run

```bash
python -m adhd_pipeline.run_all --checkpoint
```

Saves per-fold results so a disconnect doesn't waste the run.

### 6. Partial protocol run

```bash
python -m adhd_pipeline.run_all --protocols b   # subject-wise only
```

---

## Outputs

| File | Description |
|---|---|
| `results/all_results.json` | Full nested results dict |
| `results/macro_replacements.tex` | Ready-to-paste LaTeX `\renewcommand` lines |
| `results/folds.json` | Fold assignments for external replication |
| `results/figures/leakage_bars.csv` | Protocol A vs B metrics (Figure 4) |
| `results/figures/baseline_bars.csv` | Per-model accuracy (Figure 5) |
| `results/figures/roc_aucs.csv` | Per-model AUC |
| `results/figures/confusion_matrix.csv` | Pooled confusion matrix |
| `results/figures/shap_channels.csv` | Per-channel SHAP importance (Figure 8) |
| `results/shap_channel_importance.npy` | Raw channel SHAP array |
| `results/shap_band_importance.npy` | Raw band SHAP array |

---

## Paper update workflow

1. Open `main.tex`, find the macro block (~lines 50–95).
2. Replace the placeholder block with the content of `results/macro_replacements.tex`.
3. Update bar-chart coordinate blocks from the figure CSVs.
4. Recompile: `pdflatex main → bibtex main → pdflatex main → pdflatex main`.

---

## Acceptance criteria

- Protocol A CNN-BiLSTM accuracy: **85–99 %**
- Protocol B CNN-BiLSTM accuracy: **65–88 %**
- Zero subject overlap in Protocol B folds (enforced by assertion)
- All outputs written without placeholder strings

See `RUN_NOTES.md` for any deviations from the spec.

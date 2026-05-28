# Implementation Brief — Subject-Independent, Leakage-Aware CNN-BiLSTM for EEG-Based ADHD Prediction

> **Audience:** Claude Code (autonomous coding agent).
> **Goal:** Produce a fully reproducible Python pipeline that generates every numeric result, table value, and figure data point required by `main.tex`. Replace the placeholder macros in that file with real measured values at the end.
> **Authors of the paper:** Geethapriya M, Kalyanaraman P (SCOPE, VIT Vellore).

---

## 0. TL;DR for the Agent

1. Download the public Nasrabadi ADHD/Control EEG dataset (121 children, 19 channels, 128 Hz) from IEEE Dataport or Kaggle mirror.
2. Build a Python pipeline (`adhd_pipeline/`) that:
   - Preprocesses every recording identically.
   - Trains six models (3 classical + 3 deep) under **two evaluation protocols** that differ in **exactly one factor** (whether the data split respects subject identity).
   - Reports mean ± std of accuracy, precision, recall, F1, ROC-AUC across folds.
   - Computes the **leakage gap** (Protocol A − Protocol B) and tests it with Wilcoxon and paired t-test.
   - Generates SHAP attributions for the CNN-BiLSTM under the subject-wise protocol, aggregated per channel and per frequency band.
3. Write every result to `results/all_results.json` and emit a `results/macro_replacements.tex` file containing ready-to-paste LaTeX `\renewcommand` lines.
4. **Acceptance criterion:** running `python -m adhd_pipeline.run_all --quick` finishes end-to-end on a single Kaggle T4 GPU in under 90 minutes, and the full run (no `--quick`) in under 6 hours.

The agent should **not** invent results, **not** modify the paper text or macros itself, and **not** silently lower the model spec to save time. Any deliberate simplification must be flagged in `RUN_NOTES.md`.

---

## 1. Dataset

### 1.1 Source

- **Primary source:** IEEE Dataport — Nasrabadi, A. M., Allahverdy, A., Samavati, M., & Mohammadi, M. R. (2020). *EEG data for ADHD / control children.*
  DOI: `10.21227/rzfh-zn36`
  URL: `https://ieee-dataport.org/open-access/eeg-data-adhd-control-children`
  (IEEE Dataport account required, free for open-access datasets.)
- **Kaggle mirror (recommended for Kaggle/Colab runs):** search "EEG data for ADHD / control children" on Kaggle. Several user-uploaded mirrors of the same `.mat` files exist; pick the most-upvoted one and verify the file count below.

### 1.2 What you should see after download

- Two folders, typically named `ADHD_part1`, `ADHD_part2`, `Control_part1`, `Control_part2`, or similar.
- **121 `.mat` files in total: 61 ADHD + 60 Control.** Confirm this count before running anything. If you see a very different count, you have the wrong dataset.
- Each `.mat` contains one variable storing a 2D array. One dimension is `19` (channels), the other is time (varies per subject, sampling rate 128 Hz). The loader must auto-orient so that **axis 0 = time, axis 1 = channels (19)**.
- Channel order (per the dataset documentation): `Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3, F4, F7, F8, P3, P4, T5, T6, O1, O2`.

### 1.3 Expected on-disk layout after extraction

```
data/
├── ADHD/
│   ├── v1p.mat
│   ├── v2p.mat
│   └── ... (61 files)
└── Control/
    ├── v1c.mat
    ├── v2c.mat
    └── ... (60 files)
```

File names vary by mirror. The agent must **consolidate** whatever it gets into the structure above (one folder per class) before running the pipeline. Filename prefixes are not used; the parent folder determines the label.

---

## 2. Environment

### 2.1 Where to run

| Option | Recommended? | Notes |
|---|---|---|
| **Kaggle Notebook (T4 GPU)** | ✅ best | Free, 16 GB VRAM, ~8 TFLOPS, 30 GB disk. The pipeline fits comfortably. Enable internet for pip installs. |
| **Google Colab (T4 GPU)** | ✅ good | Free tier works; runtime disconnects after ~3 h of idle, so use `--checkpoint` (see §4.5). |
| **Local GTX 1050 (4 GB VRAM)** | ⚠ marginal | Works only with reduced batch size and reduced SHAP sample size; see §4.6. Expect 4–6× longer runs. |
| **CPU only** | ❌ no | Deep model training will take ~24 h. Don't. |

### 2.2 Python and packages

- **Python 3.10 or 3.11** (TensorFlow 2.15+ requires this; do not use 3.12 yet).
- **CUDA:** matches the Kaggle/Colab default; do not pin a CUDA version manually.

Create `requirements.txt`:

```
numpy>=1.24,<2.0
scipy>=1.10
scikit-learn>=1.3
pandas>=2.0
matplotlib>=3.7
tensorflow>=2.15,<2.17
shap>=0.44
mne>=1.6
tqdm
joblib
```

Pin `numpy<2` because SHAP and older TF wheels are not yet fully compatible with NumPy 2.x.

### 2.3 Reproducibility seeds

Set, in this order, at the start of every entry point:

```python
import os, random, numpy as np, tensorflow as tf
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
```

Save the resolved fold assignments to `results/folds.json` so an external reviewer can replicate them.

---

## 3. Repository Layout to Create

```
.
├── README.md                    ← short user-facing instructions (you write this)
├── RUN_NOTES.md                 ← log any deviations, simplifications, surprises
├── requirements.txt
├── data/                        ← (gitignored) populated by §1
├── results/                     ← (gitignored) all outputs land here
│   ├── all_results.json
│   ├── macro_replacements.tex   ← ready-to-paste LaTeX
│   ├── folds.json
│   ├── shap_channel_importance.npy
│   ├── shap_band_importance.npy
│   └── figures/
│       ├── leakage_bars.csv
│       ├── baseline_bars.csv
│       ├── roc_aucs.csv
│       ├── confusion_matrix.csv
│       └── shap_channels.csv
└── adhd_pipeline/
    ├── __init__.py
    ├── config.py                ← all hyper-parameters in one place
    ├── data_io.py               ← .mat loading, folder resolution, subject indexing
    ├── preprocessing.py         ← band-pass, notch, z-score (train-only), epoching
    ├── features.py              ← band-power features for classical baselines
    ├── splits.py                ← segment-wise, subject-wise, LOSO splitters
    ├── models_classical.py      ← LR, SVM (RBF), RandomForest
    ├── models_deep.py           ← CNN-only, LSTM-only, CNN-BiLSTM (Keras)
    ├── evaluate.py              ← metric computation, per-fold loop, paired tests
    ├── explain.py               ← SHAP attribution, channel/band aggregation
    ├── report.py                ← writes JSON and macro_replacements.tex
    └── run_all.py               ← CLI entry point
```

---

## 4. File-by-File Specification

### 4.1 `config.py`

Single source of truth. The paper's Methods section quotes these numbers, so they must match exactly:

```python
FS          = 128
BANDPASS    = (0.5, 50.0)
NOTCH_FREQ  = 50.0
EPOCH_SEC   = 4
EPOCH_LEN   = EPOCH_SEC * FS              # 512 samples
N_CHANNELS  = 19
CHANNEL_ORDER = ["Fz","Cz","Pz","C3","T3","C4","T4","Fp1","Fp2",
                 "F3","F4","F7","F8","P3","P4","T5","T6","O1","O2"]
BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),
         "beta":(13,30),"gamma":(30,50)}
K_FOLDS     = 10
DL_EPOCHS   = 60
DL_BATCH    = 64        # on GTX 1050, drop to 16; see §4.6
DL_LR       = 1e-3
DL_PATIENCE = 10
DL_DROPOUT  = 0.3
RANDOM_SEED = 42
```

### 4.2 `data_io.py`

- `discover_dataset(root) -> (adhd_dir, control_dir)`: tolerant to mirror naming. If folders are named `ADHD_part1`, `ADHD_part2`, etc., consolidate into the expected layout (or treat them all as one class).
- `load_subjects(folder, label) -> list[(signal, label, subject_id)]`:
  - Open each `.mat` with `scipy.io.loadmat`.
  - Skip `__*__` keys.
  - Find the first 2D numeric array; if `min(shape) >= 2`, orient so axis 1 is the channel axis with size `N_CHANNELS`. If a file does not have 19 channels in either orientation, log a warning and skip it (record skipped files in `RUN_NOTES.md`).
  - Subject ID = filename stem.
- Assert at the end of loading: **61 ADHD + 60 Control**, else raise with a clear error message.

### 4.3 `preprocessing.py`

- `bandpass_notch(signal)`: 4th-order Butterworth band-pass `[0.5, 50] Hz`, zero-phase (`scipy.signal.filtfilt`), then `iirnotch` at 50 Hz, Q=30. Apply per channel.
- `epoch(signal) -> ndarray (n_epochs, EPOCH_LEN, N_CHANNELS)`: non-overlapping, drop the tail.
- `build_dataset(subjects) -> (X, y, groups)`: arrays. `groups` is an integer subject index aligned to `X`.
- `zscore_train_only(X_train, X_test) -> (X_train_n, X_test_n)`: compute mean and std from `X_train` over axes `(0, 1)`, apply to both. **Never** compute statistics from the test set.

### 4.4 `features.py`

- `bandpower_features(X) -> ndarray (n_epochs, N_CHANNELS * 5)`: for each epoch and each channel, compute Welch PSD with `nperseg=min(EPOCH_LEN, 256)`, then the **relative** power in each of the five bands (band power / total power). Flatten in channel-major order so columns can be mapped back to (channel, band) for SHAP reporting.

### 4.5 `splits.py`

Three functions, each returning a list of `(train_idx, test_idx)` tuples:

1. `folds_segment_wise(y, groups)` — stratified `K_FOLDS` at the **epoch level**. Subjects can be on both sides.
2. `folds_subject_wise(y, groups)` — stratified `K_FOLDS` at the **subject level**: take the unique subject IDs, stratify by each subject's label, then expand to epoch indices.
3. `folds_loso(y, groups)` — `LeaveOneGroupOut`.

Persist all three to `results/folds.json` after generation.

### 4.6 `models_deep.py`

CNN-BiLSTM architecture (must exactly match Methods §3.4 of the paper):

- Input: `(EPOCH_LEN, N_CHANNELS) = (512, 19)`.
- `Conv1D(64, kernel=7, padding="same")` → `BatchNorm` → `ReLU` → `MaxPool1D(2)` → `Dropout(0.3)`.
- `Conv1D(128, kernel=5, padding="same")` → `BatchNorm` → `ReLU` → `MaxPool1D(2)` → `Dropout(0.3)`.
- `Bidirectional(LSTM(64, return_sequences=True))`.
- `Bidirectional(LSTM(32))`.
- `Dense(64, activation="relu")` → `Dropout(0.3)`.
- `Dense(1, activation="sigmoid")`.

Optimiser: `Adam(lr=1e-3)`. Loss: `binary_crossentropy`. EarlyStopping(monitor=`val_loss`, patience=10, restore_best_weights=True). Use 15% of the training fold as the validation split.

Also provide `make_cnn_only` (same Conv stack + GlobalAveragePooling + Dense head) and `make_lstm_only` (same recurrent stack + Dense head) for the baselines, built from the same sub-components so the comparison cleanly isolates the contribution of combining them.

**GTX 1050 fallback (if you actually run on the 1050):** reduce `DL_BATCH` to 16 and `BiLSTM` units to (32, 16). Note this in `RUN_NOTES.md` and **do not** edit the paper text; the paper's architecture description is the one we built and report.

### 4.7 `evaluate.py`

- `score(y_true, y_pred, y_prob) -> dict` with keys `accuracy` (in %), `precision`, `recall`, `f1`, `roc_auc`. Use `zero_division=0` everywhere.
- `evaluate_classical(name, builder, Xfeat, y, splits)`: standard fit/predict loop, returns one score dict per fold.
- `evaluate_deep(name, builder, X, y, splits, collect_predictions=False)`: per-fold loop with `zscore_train_only`, fresh model each fold, early stopping, optional accumulation of `(y_true, y_pred)` for confusion-matrix pooling.
- `summarise(fold_scores) -> dict[metric, (mean, std)]`.
- `paired_significance(acc_a, acc_b) -> dict` with Wilcoxon and paired t statistics and p-values.

### 4.8 `explain.py`

- Train the CNN-BiLSTM on fold 0 of the **subject-wise** split.
- Background set: 100 random training epochs.
- Sample set: up to 200 test epochs (≤50 on GTX 1050).
- Use `shap.GradientExplainer` (works with TF/Keras and is fast enough on T4).
- Aggregate `|SHAP|` two ways:
  - **Per channel:** mean of `|SHAP|` over (samples, time).
  - **Per band:** for each band, mask the time-frequency contribution by approximating with a band-pass filter applied to the input, then re-explain. If that is too slow on the 1050, fall back to: compute Welch power per band on the explained samples and report the correlation between band power and the SHAP-aggregated saliency. State which method was used in `RUN_NOTES.md`.
- Save `shap_channel_importance.npy`, `shap_band_importance.npy`, and a CSV `shap_channels.csv` with columns `channel_name, mean_abs_shap`.

### 4.9 `report.py`

After all experiments complete, write **two** files. The first is `all_results.json` — the full nested dict. The second, `macro_replacements.tex`, contains LaTeX lines that **exactly** redefine the placeholder macros in `main.tex`. Format:

```latex
% Auto-generated -- paste into main.tex to replace the placeholder block.
\renewcommand{\AccA}{92.4}   \renewcommand{\AccAsd}{1.8}
\renewcommand{\PrecA}{0.92}  \renewcommand{\RecA}{0.93}
\renewcommand{\FoneA}{0.92}  \renewcommand{\AucA}{0.96}
\renewcommand{\AccB}{79.6}   \renewcommand{\AccBsd}{4.2}
... etc ...
\renewcommand{\Gap}{12.8}
\renewcommand{\GapP}{0.001}
\renewcommand{\GapT}{0.001}
\renewcommand{\TP}{485}  \renewcommand{\FN}{125}
\renewcommand{\FP}{120}  \renewcommand{\TN}{470}
\renewcommand{\Params}{1.18}
\renewcommand{\TrainMin}{14}
\renewcommand{\InferMs}{8}
```

Round accuracies to one decimal, probabilistic metrics to two decimals, p-values to three significant figures. **Never** emit fabricated numbers — every value must come from a measurement in this run.

Also emit CSVs of the data behind each figure so the bar charts and ROC curves in `main.tex` can be updated by editing literal coordinates:

- `leakage_bars.csv`: rows `metric, protocol_A, protocol_B`.
- `baseline_bars.csv`: rows `model, accuracy_mean, accuracy_std`.
- `roc_aucs.csv`: rows `model, auc`.
- `confusion_matrix.csv`: 2x2 grid.
- `shap_channels.csv`: as above.

### 4.10 `run_all.py`

CLI flags:

- `--data-dir PATH` (default `./data`)
- `--out-dir PATH` (default `./results`)
- `--quick` — reduces `DL_EPOCHS` to 10, `K_FOLDS` to 5, and limits SHAP samples to 50. Used for smoke testing the full pipeline in ~10 minutes. Results from a `--quick` run **must not** be pasted into the paper; they are for debugging only. The agent should write `QUICK_RUN=True` into `macro_replacements.tex` as a comment when this flag is used.
- `--protocols a,b,loso` — comma-separated subset for partial runs.
- `--checkpoint` — save fold-level results incrementally so a Colab disconnect doesn't lose the run.

The driver prints a clean per-protocol summary at the end:

```
PROTOCOL B (subject-wise)
  LogisticRegression  acc=68.2 ± 5.1  auc=0.72
  SVM                 acc=71.4 ± 4.8  auc=0.76
  ...
  CNN-BiLSTM          acc=79.6 ± 4.2  auc=0.85

LEAKAGE GAP (CNN-BiLSTM)
  accuracy  A=92.4  B=79.6  gap=12.8
  Wilcoxon  stat=...  p=0.001
  Paired t  stat=...  p=0.001
```

---

## 5. Kaggle / Colab Quick-Start

### 5.1 Kaggle (recommended)

1. Create a new notebook, attach the dataset (search "ADHD EEG" — pick the mirror with 121 `.mat` files), enable **Internet** in the side panel, set Accelerator to **GPU T4 x1**.
2. In a single cell:
   ```bash
   !git clone <your-repo-url> work && cd work && pip install -q -r requirements.txt
   ```
3. Adjust paths to point at `/kaggle/input/<dataset-slug>/` and run:
   ```bash
   !cd work && python -m adhd_pipeline.run_all --data-dir /kaggle/input/<slug> --out-dir /kaggle/working/results
   ```
4. After it finishes, download `results/macro_replacements.tex` and `results/figures/*.csv`.

### 5.2 Colab

Same flow; mount Google Drive for persistence, install requirements, set runtime → T4 GPU. Use `--checkpoint` so an idle disconnect after ~3 h doesn't waste the run.

### 5.3 Local GTX 1050

Only if Kaggle/Colab are unavailable. Set `DL_BATCH=16`, reduce BiLSTM units as in §4.6, expect 4–6 h. Monitor `nvidia-smi`; if you OOM at the CNN-BiLSTM step, halve the batch again.

---

## 6. Acceptance Criteria

The agent's run is considered successful when **all** of the following hold:

1. `data/` contains 61 ADHD + 60 Control `.mat` files, confirmed by `data_io.discover_dataset`.
2. `results/all_results.json` exists, is valid JSON, and contains entries for all three protocols × all six models.
3. `results/macro_replacements.tex` is generated and contains no placeholder strings (`XX.X`, `TODO`, etc.).
4. The CNN-BiLSTM **subject-wise accuracy is in the range 65 % – 88 %** and the **segment-wise accuracy is in the range 85 % – 99 %**. If either is outside this range, treat it as a likely bug (probably a data-loading orientation issue or a leaky split) and investigate before reporting. Document any genuine anomaly in `RUN_NOTES.md`.
5. The leakage-gap p-value from the paired test is reported (whether or not it is significant).
6. SHAP outputs exist and the top-5 channels by mean `|SHAP|` are saved to `shap_channels.csv`.
7. The `--quick` smoke test finishes without error end-to-end before the full run is launched.

---

## 7. After the Run — Paper Update Workflow

This part is **manual**, not for the agent:

1. Open `main.tex`, find the macro block (around lines 50–95).
2. Replace each `\newcommand{\X}{...}` line with the corresponding `\renewcommand` value from `results/macro_replacements.tex`. The cleanest path is to delete the placeholder block and paste the generated block in its place.
3. Update the two bar-chart figure coordinate blocks (Figure 4 leakage gap, Figure 5 baseline comparison) from `leakage_bars.csv` and `baseline_bars.csv`.
4. Update the SHAP chart coordinates (Figure 8) from `shap_channels.csv`.
5. Recompile: `pdflatex main → bibtex main → pdflatex main → pdflatex main`.
6. Read the paper end-to-end once. Pay attention to the Discussion section: a couple of sentences interpret the size of the leakage gap and the ordering of the baselines. If your real numbers are very different from the placeholders, rewrite those interpretive sentences honestly.

---

## 8. Things the Agent Must Not Do

- **Do not** invent or fill in any number that wasn't measured in this run.
- **Do not** edit `main.tex` directly. Produce `macro_replacements.tex` and let the human apply it.
- **Do not** swap the architecture or hyper-parameters without recording the change in `RUN_NOTES.md` and stating that the paper's Methods section will need a matching edit.
- **Do not** use overlapping epochs to "boost" the sample count. The paper explicitly argues for non-overlapping epochs as the conservative choice; departing from this silently would invalidate the headline finding.
- **Do not** mix the test fold into preprocessing statistics. `zscore_train_only` is named the way it is for a reason.

---

## 9. Sanity Checks to Run Before Reporting

Before considering the job done, the agent should print and inspect:

1. **Data shape report:** total epochs, epochs per class, epochs per subject (min, max, median).
2. **Class balance per fold** for both protocols (should stay close to 50/50).
3. **Subject overlap check** for Protocol B: assert `set(groups[train_idx]).intersection(set(groups[test_idx])) == set()` for every fold. If this assertion ever fails, the entire run is invalid.
4. **A single deterministic re-run** of fold 0 of Protocol B with the same seed should reproduce accuracy to within ±0.5 percentage points.

If any check fails, fix the bug, not the metric.

---

End of brief.
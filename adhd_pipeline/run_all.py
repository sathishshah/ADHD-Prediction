"""
Main CLI entry point for the ADHD EEG pipeline.

Usage:
    python -m adhd_pipeline.run_all [options]

Options:
    --data-dir PATH       Root of the dataset (default: ./data)
    --out-dir PATH        Output directory (default: ./results)
    --quick               Smoke-test mode: 10 epochs, 5 folds, 50 SHAP samples
    --protocols a,b,loso  Comma-separated subset of protocols to run
    --checkpoint          Save fold results incrementally
    --no-shap             Skip SHAP computation
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# ── Reproducibility — must happen before any TF import ──────────────────────
os.environ["PYTHONHASHSEED"]    = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

# ── Local imports ─────────────────────────────────────────────────────────────
from . import config
from .data_io       import load_all
from .preprocessing import build_dataset
from .features      import bandpower_features
from .splits        import folds_segment_wise, folds_subject_wise, folds_loso, save_folds
from .models_classical import CLASSICAL_MODELS
from .models_deep      import DEEP_MODELS
from .evaluate         import (
    evaluate_classical, evaluate_deep, summarise, paired_significance,
)
from .explain  import compute_shap
from .report   import write_results, print_protocol_summary


def _parse_args():
    parser = argparse.ArgumentParser(description="ADHD EEG Pipeline")
    parser.add_argument("--data-dir",   default="./data",    help="Dataset root")
    parser.add_argument("--out-dir",    default="./results", help="Output directory")
    parser.add_argument("--quick",      action="store_true", help="Smoke-test mode")
    parser.add_argument("--protocols",  default="a,b,loso",  help="Protocols to run")
    parser.add_argument("--checkpoint", action="store_true", help="Save per-fold results")
    parser.add_argument("--no-shap",    action="store_true", help="Skip SHAP")
    return parser.parse_args()


def _apply_quick_overrides():
    config.DL_EPOCHS = 10
    config.K_FOLDS   = 5
    print("[QUICK MODE] DL_EPOCHS=10, K_FOLDS=5, SHAP samples=50")


def _sanity_checks(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    print("\n── Sanity checks ──────────────────────────────────────────────")
    n_total = len(y)
    n_adhd  = (y == 1).sum()
    n_ctrl  = (y == 0).sum()
    print(f"  Total epochs : {n_total}  (ADHD={n_adhd}, Control={n_ctrl})")

    epochs_per_subj = [int((groups == s).sum()) for s in np.unique(groups)]
    print(f"  Epochs/subject: min={min(epochs_per_subj)}  "
          f"max={max(epochs_per_subj)}  median={int(np.median(epochs_per_subj))}")
    print("───────────────────────────────────────────────────────────────\n")


def _check_subject_overlap(splits, groups, label: str) -> None:
    for fold_idx, (tr, te) in enumerate(splits):
        overlap = set(groups[tr]) & set(groups[te])
        if overlap:
            raise AssertionError(
                f"Subject overlap in {label} fold {fold_idx}: {overlap}"
            )
    print(f"  Subject overlap check PASSED for {label}")


def _pool_confusion_matrix(predictions):
    """Aggregate pooled confusion matrix from list of (y_true, y_pred, y_prob)."""
    tp = fp = fn = tn = 0
    for y_true, y_pred, _ in predictions:
        tp += int(((y_true == 1) & (y_pred == 1)).sum())
        fn += int(((y_true == 1) & (y_pred == 0)).sum())
        fp += int(((y_true == 0) & (y_pred == 1)).sum())
        tn += int(((y_true == 0) & (y_pred == 0)).sum())
    return [[tp, fn], [fp, tn]]


def _count_params(builder) -> float:
    """Return parameter count of a model in millions."""
    model = builder()
    total = model.count_params()
    tf.keras.backend.clear_session()
    return total / 1e6


def _checkpoint_path(out_dir: Path, protocol: str, model_name: str) -> Path:
    return out_dir / f"checkpoint_{protocol}_{model_name}.json"


def _load_checkpoint(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_checkpoint(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def main():
    args = _parse_args()

    if args.quick:
        _apply_quick_overrides()

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    protocols_to_run = {p.strip().lower() for p in args.protocols.split(",")}

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("Loading dataset …")
    subjects = load_all(args.data_dir)

    # ── 2. Preprocess and epoch ───────────────────────────────────────────────
    print("Preprocessing and epoching …")
    X, y, groups = build_dataset(subjects)
    _sanity_checks(X, y, groups)

    # ── 3. Build splits ───────────────────────────────────────────────────────
    print("Building CV splits …")
    splits_a    = folds_segment_wise(y, groups)
    splits_b    = folds_subject_wise(y, groups)
    splits_loso = folds_loso(y, groups)

    _check_subject_overlap(splits_b,    groups, "subject-wise")
    _check_subject_overlap(splits_loso, groups, "LOSO")

    save_folds(splits_a, splits_b, splits_loso, str(out_dir))

    # ── 4. Band-power features for classical models ───────────────────────────
    print("Extracting band-power features …")
    X_feat = bandpower_features(X)

    # ── 5. Run experiments ────────────────────────────────────────────────────
    all_results: dict = {}
    cnn_acc_a: list[float] = []
    cnn_acc_b: list[float] = []

    protocol_map = {
        "a":    ("protocol_a", splits_a,    "PROTOCOL A (segment-wise, leaky)"),
        "b":    ("protocol_b", splits_b,    "PROTOCOL B (subject-wise, clean)"),
        "loso": ("loso",       splits_loso, "PROTOCOL LOSO"),
    }

    for proto_key, (result_key, splits, label) in protocol_map.items():
        if proto_key not in protocols_to_run:
            continue

        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        all_results[result_key] = {}

        # Classical models
        for name, builder in CLASSICAL_MODELS.items():
            ck_path = _checkpoint_path(out_dir, proto_key, name)
            cached  = _load_checkpoint(ck_path) if args.checkpoint else None
            if cached:
                print(f"  [checkpoint] Loading {name} ({proto_key})")
                fold_scores = cached
            else:
                print(f"\n  {name}")
                fold_scores = evaluate_classical(name, builder, X_feat, y, splits)
                if args.checkpoint:
                    _save_checkpoint(ck_path, fold_scores)

            all_results[result_key][name] = {
                "fold_scores": fold_scores,
                "summary":     summarise(fold_scores),
            }

        # Deep models
        collect_preds = (proto_key == "b")   # collect for confusion matrix
        for name, builder in DEEP_MODELS.items():
            ck_path = _checkpoint_path(out_dir, proto_key, name)
            cached  = _load_checkpoint(ck_path) if args.checkpoint else None
            if cached:
                print(f"  [checkpoint] Loading {name} ({proto_key})")
                fold_scores, predictions = cached["fold_scores"], None
            else:
                print(f"\n  {name}")
                fold_scores, predictions = evaluate_deep(
                    name, builder, X, y, splits,
                    collect_predictions=collect_preds,
                )
                if args.checkpoint:
                    _save_checkpoint(ck_path, {"fold_scores": fold_scores})

            all_results[result_key][name] = {
                "fold_scores": fold_scores,
                "summary":     summarise(fold_scores),
            }

            if name == "CNN_BiLSTM":
                accs = [s["accuracy"] for s in fold_scores]
                if proto_key == "a":
                    cnn_acc_a = accs
                elif proto_key == "b":
                    cnn_acc_b = accs
                    if collect_preds and predictions:
                        all_results["confusion_matrix"] = _pool_confusion_matrix(predictions)

    # ── 6. Statistical significance of leakage gap ────────────────────────────
    if cnn_acc_a and cnn_acc_b and len(cnn_acc_a) == len(cnn_acc_b):
        print("\nComputing leakage gap significance …")
        all_results["significance"] = paired_significance(cnn_acc_a, cnn_acc_b)
    else:
        all_results["significance"] = {}

    # ── 7. Model parameter count and timing ───────────────────────────────────
    cnn_params = _count_params(DEEP_MODELS["CNN_BiLSTM"])
    all_results["params"] = cnn_params

    if "protocol_b" in all_results and "CNN_BiLSTM" in all_results["protocol_b"]:
        fold_scores_b = all_results["protocol_b"]["CNN_BiLSTM"]["fold_scores"]
        train_sec = np.sum([s.get("train_sec", 0) for s in fold_scores_b])
        all_results["timing"] = {
            "train_min": train_sec / 60,
            "infer_ms":  0,   # updated below
        }

    # Measure inference latency (single epoch, 10 warm-up + 100 timed)
    try:
        print("Measuring inference latency …")
        model_for_timing = DEEP_MODELS["CNN_BiLSTM"]()
        dummy = np.zeros((1, config.EPOCH_LEN, config.N_CHANNELS), dtype=np.float32)
        for _ in range(10):
            model_for_timing.predict(dummy, verbose=0)
        t0 = time.time()
        for _ in range(100):
            model_for_timing.predict(dummy, verbose=0)
        infer_ms = (time.time() - t0) / 100 * 1000
        if "timing" in all_results:
            all_results["timing"]["infer_ms"] = infer_ms
        tf.keras.backend.clear_session()
    except Exception as exc:
        print(f"  Inference timing failed: {exc}")

    # ── 8. SHAP ───────────────────────────────────────────────────────────────
    if not args.no_shap and "b" in protocols_to_run:
        n_shap = 50 if args.quick else 200
        print(f"\nComputing SHAP (n_samples={n_shap}) …")
        try:
            compute_shap(X, y, splits_b, str(out_dir), n_samples=n_shap)
        except Exception as exc:
            print(f"  SHAP computation failed: {exc}")

    # ── 9. Acceptance range checks ────────────────────────────────────────────
    print("\n── Acceptance range checks ────────────────────────────────────")
    if "protocol_a" in all_results and "CNN_BiLSTM" in all_results["protocol_a"]:
        acc_a = all_results["protocol_a"]["CNN_BiLSTM"]["summary"]["accuracy"][0]
        if not (85 <= acc_a <= 99):
            print(f"  WARNING: Protocol A CNN-BiLSTM acc={acc_a:.1f} outside [85, 99].")
        else:
            print(f"  Protocol A acc={acc_a:.1f} ✓")
    if "protocol_b" in all_results and "CNN_BiLSTM" in all_results["protocol_b"]:
        acc_b = all_results["protocol_b"]["CNN_BiLSTM"]["summary"]["accuracy"][0]
        if not (65 <= acc_b <= 88):
            print(f"  WARNING: Protocol B CNN-BiLSTM acc={acc_b:.1f} outside [65, 88].")
        else:
            print(f"  Protocol B acc={acc_b:.1f} ✓")
    print("───────────────────────────────────────────────────────────────")

    # ── 10. Write outputs ─────────────────────────────────────────────────────
    write_results(all_results, str(out_dir), quick_run=args.quick)
    print_protocol_summary(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()

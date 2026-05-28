"""Per-fold evaluation loops, metric computation, and significance testing."""

import time
from typing import Callable

import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
import tensorflow as tf

from . import config
from .preprocessing import zscore_train_only

SplitList = list[tuple[np.ndarray, np.ndarray]]


def score(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return accuracy (%), precision, recall, F1, ROC-AUC."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def evaluate_classical(
    name: str,
    builder: Callable,
    X_feat: np.ndarray,
    y: np.ndarray,
    splits: SplitList,
) -> list[dict]:
    """Fit/predict loop for classical sklearn models. Returns one score dict per fold."""
    fold_scores = []
    for fold_idx, (tr, te) in enumerate(splits):
        model = builder()
        model.fit(X_feat[tr], y[tr])
        y_pred = model.predict(X_feat[te])
        y_prob = model.predict_proba(X_feat[te])[:, 1]
        s = score(y[te], y_pred, y_prob)
        s["fold"] = fold_idx
        fold_scores.append(s)
        print(f"  {name} fold {fold_idx:2d}: acc={s['accuracy']:.1f}  auc={s['roc_auc']:.3f}")
    return fold_scores


def evaluate_deep(
    name: str,
    builder: Callable,
    X: np.ndarray,
    y: np.ndarray,
    splits: SplitList,
    collect_predictions: bool = False,
) -> tuple[list[dict], list[tuple] | None]:
    """
    Per-fold training loop for Keras models.
    Returns (fold_scores, predictions_per_fold).
    predictions_per_fold is a list of (y_true, y_pred, y_prob) if
    collect_predictions is True, else None.
    """
    fold_scores  = []
    predictions  = [] if collect_predictions else None
    train_times  = []

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.DL_PATIENCE,
        restore_best_weights=True,
    )

    for fold_idx, (tr, te) in enumerate(splits):
        X_tr, X_te = zscore_train_only(X[tr], X[te])

        model = builder()

        t0 = time.time()
        model.fit(
            X_tr, y[tr],
            epochs=config.DL_EPOCHS,
            batch_size=config.DL_BATCH,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0,
        )
        train_times.append(time.time() - t0)

        y_prob = model.predict(X_te, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        s = score(y[te], y_pred, y_prob)
        s["fold"]       = fold_idx
        s["train_sec"]  = train_times[-1]
        fold_scores.append(s)
        print(f"  {name} fold {fold_idx:2d}: acc={s['accuracy']:.1f}  auc={s['roc_auc']:.3f}")

        if collect_predictions:
            predictions.append((y[te], y_pred, y_prob))

        tf.keras.backend.clear_session()

    return fold_scores, predictions


def summarise(fold_scores: list[dict]) -> dict:
    """Return {metric: (mean, std)} across folds."""
    metrics = [k for k in fold_scores[0] if k not in ("fold", "train_sec")]
    summary = {}
    for m in metrics:
        vals = np.array([s[m] for s in fold_scores])
        summary[m] = (float(vals.mean()), float(vals.std()))
    return summary


def paired_significance(acc_a: list[float], acc_b: list[float]) -> dict:
    """
    Wilcoxon signed-rank and paired t-test between two lists of per-fold accuracies.
    Returns dict with both test statistics and p-values.
    """
    a = np.array(acc_a)
    b = np.array(acc_b)

    try:
        w_stat, w_p = wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    t_stat, t_p = ttest_rel(a, b)

    return {
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p":    float(w_p),
        "ttest_stat":    float(t_stat),
        "ttest_p":       float(t_p),
    }

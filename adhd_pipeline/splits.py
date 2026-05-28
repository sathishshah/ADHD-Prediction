"""Cross-validation splitters: segment-wise, subject-wise, LOSO."""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from . import config

SplitList = list[tuple[np.ndarray, np.ndarray]]


def folds_segment_wise(y: np.ndarray, groups: np.ndarray) -> SplitList:
    """
    Stratified K-fold at the epoch level.
    Subjects can appear on both sides of the split (leaky protocol).
    """
    skf = StratifiedKFold(
        n_splits=config.K_FOLDS, shuffle=True,
        random_state=config.RANDOM_SEED,
    )
    return list(skf.split(np.zeros(len(y)), y))


def folds_subject_wise(y: np.ndarray, groups: np.ndarray) -> SplitList:
    """
    Stratified K-fold at the subject level.
    Unique subjects are stratified by their majority label, then all
    epochs belonging to a subject are moved together.
    """
    unique_subjects = np.unique(groups)

    # Each subject's label = majority label of its epochs
    subj_labels = np.array(
        [int(np.round(y[groups == s].mean())) for s in unique_subjects]
    )

    skf = StratifiedKFold(
        n_splits=config.K_FOLDS, shuffle=True,
        random_state=config.RANDOM_SEED,
    )

    folds: SplitList = []
    for train_subj_idx, test_subj_idx in skf.split(unique_subjects, subj_labels):
        train_subjects = unique_subjects[train_subj_idx]
        test_subjects  = unique_subjects[test_subj_idx]

        train_idx = np.where(np.isin(groups, train_subjects))[0]
        test_idx  = np.where(np.isin(groups, test_subjects))[0]

        # Sanity: no subject overlap
        assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, \
            "Subject overlap detected in subject-wise split!"

        folds.append((train_idx, test_idx))

    return folds


def folds_loso(y: np.ndarray, groups: np.ndarray) -> SplitList:
    """Leave-One-Subject-Out."""
    logo = LeaveOneGroupOut()
    return list(logo.split(np.zeros(len(y)), y, groups))


def save_folds(
    folds_a: SplitList,
    folds_b: SplitList,
    folds_loso: SplitList,
    out_dir: str,
) -> None:
    """Persist all fold assignments to results/folds.json."""
    def _serialise(folds):
        return [
            {"train": tr.tolist(), "test": te.tolist()}
            for tr, te in folds
        ]

    payload = {
        "segment_wise": _serialise(folds_a),
        "subject_wise": _serialise(folds_b),
        "loso":         _serialise(folds_loso),
    }
    out_path = Path(out_dir) / "folds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Fold assignments saved to {out_path}")

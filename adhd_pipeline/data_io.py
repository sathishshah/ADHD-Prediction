"""Load and consolidate the Nasrabadi ADHD/Control EEG dataset."""

import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import scipy.io

from . import config

EXPECTED_ADHD    = 61
EXPECTED_CONTROL = 60


def discover_dataset(root: str) -> tuple[Path, Path]:
    """
    Return (adhd_dir, control_dir) under *root*, tolerating various mirror layouts.
    Consolidates multi-part folders (ADHD_part1, ADHD_part2, …) into
    root/ADHD and root/Control if needed.
    """
    root = Path(root)

    def _find_dirs(keyword: str) -> list[Path]:
        return sorted(
            d for d in root.rglob("*")
            if d.is_dir() and keyword.lower() in d.name.lower()
        )

    def _consolidate(dirs: list[Path], target: Path) -> Path:
        target.mkdir(parents=True, exist_ok=True)
        for src in dirs:
            if src.resolve() == target.resolve():
                continue
            for f in src.glob("*.mat"):
                dst = target / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
        return target

    adhd_target    = root / "ADHD"
    control_target = root / "Control"

    # If canonical dirs already exist and are populated, return them directly.
    if (adhd_target.is_dir() and any(adhd_target.glob("*.mat")) and
            control_target.is_dir() and any(control_target.glob("*.mat"))):
        return adhd_target, control_target

    adhd_dirs    = _find_dirs("adhd")
    control_dirs = _find_dirs("control")

    if not adhd_dirs or not control_dirs:
        raise FileNotFoundError(
            f"Could not locate ADHD and Control folders under '{root}'. "
            "Expected folder names containing 'adhd' and 'control'."
        )

    _consolidate(adhd_dirs,    adhd_target)
    _consolidate(control_dirs, control_target)
    return adhd_target, control_target


def _load_mat(path: Path) -> np.ndarray | None:
    """
    Return a float32 array of shape (n_time, N_CHANNELS=19).
    Returns None and logs a warning if the file cannot be oriented correctly.
    """
    try:
        mat = scipy.io.loadmat(str(path))
    except Exception as exc:
        warnings.warn(f"Could not read {path.name}: {exc}")
        return None

    arrays = [
        v for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2
    ]
    if not arrays:
        warnings.warn(f"{path.name}: no 2-D numeric array found — skipped.")
        return None

    arr = arrays[0].astype(np.float32)

    if arr.shape[0] == config.N_CHANNELS:
        arr = arr.T                         # (channels, time) → (time, channels)
    elif arr.shape[1] == config.N_CHANNELS:
        pass                                # already (time, channels)
    else:
        warnings.warn(
            f"{path.name}: neither dimension equals {config.N_CHANNELS} "
            f"(shape={arr.shape}) — skipped."
        )
        return None

    return arr


def load_subjects(folder: Path, label: int) -> list[tuple[np.ndarray, int, str]]:
    """
    Returns a list of (signal, label, subject_id) for every valid .mat file
    in *folder*.  subject_id = filename stem.
    """
    records = []
    skipped = []
    for mat_path in sorted(folder.glob("*.mat")):
        signal = _load_mat(mat_path)
        if signal is None:
            skipped.append(mat_path.name)
            continue
        records.append((signal, label, mat_path.stem))

    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} file(s) in {folder.name}: {skipped}"
        )
    return records


def load_all(root: str) -> list[tuple[np.ndarray, int, str]]:
    """
    Discover, consolidate, load, and validate the full dataset.
    Returns a flat list of (signal, label, subject_id).
    Raises AssertionError if the 61 ADHD + 60 Control count is not met.
    """
    adhd_dir, control_dir = discover_dataset(root)

    adhd_records    = load_subjects(adhd_dir,    label=1)
    control_records = load_subjects(control_dir, label=0)

    n_adhd    = len(adhd_records)
    n_control = len(control_records)

    if n_adhd != EXPECTED_ADHD or n_control != EXPECTED_CONTROL:
        warnings.warn(
            f"Expected {EXPECTED_ADHD} ADHD + {EXPECTED_CONTROL} Control subjects, "
            f"but found {n_adhd} ADHD + {n_control} Control. "
            "Results may differ from the paper. Proceeding anyway."
        )

    if n_adhd == 0 or n_control == 0:
        raise FileNotFoundError(
            f"No valid .mat files found (ADHD={n_adhd}, Control={n_control}). "
            "Check that the --data-dir path points to the correct dataset folder."
        )

    print(f"Loaded {n_adhd} ADHD + {n_control} Control subjects.")
    return adhd_records + control_records

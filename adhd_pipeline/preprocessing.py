"""Signal filtering, epoching, and normalization."""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from . import config


def bandpass_notch(signal: np.ndarray) -> np.ndarray:
    """
    Apply 4th-order Butterworth bandpass [0.5, 50] Hz (zero-phase) then
    iirnotch at 50 Hz (Q=30) to each channel independently.
    signal: (n_time, N_CHANNELS)
    """
    nyq = config.FS / 2.0
    low, high = config.BANDPASS[0] / nyq, config.BANDPASS[1] / nyq
    b_bp, a_bp = butter(4, [low, high], btype="band")

    b_notch, a_notch = iirnotch(config.NOTCH_FREQ / nyq, Q=30)

    out = np.empty_like(signal)
    for ch in range(signal.shape[1]):
        x = filtfilt(b_bp, a_bp, signal[:, ch])
        x = filtfilt(b_notch, a_notch, x)
        out[:, ch] = x
    return out


def epoch(signal: np.ndarray) -> np.ndarray:
    """
    Split (n_time, N_CHANNELS) into non-overlapping epochs.
    Returns (n_epochs, EPOCH_LEN, N_CHANNELS); tail is dropped.
    """
    n_time = signal.shape[0]
    n_epochs = n_time // config.EPOCH_LEN
    trimmed = signal[: n_epochs * config.EPOCH_LEN]
    return trimmed.reshape(n_epochs, config.EPOCH_LEN, config.N_CHANNELS)


def build_dataset(
    subjects: list[tuple[np.ndarray, int, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess all subjects and epoch them.
    Returns:
        X      (n_epochs, EPOCH_LEN, N_CHANNELS)  float32
        y      (n_epochs,)                         int
        groups (n_epochs,)                         int  — subject index
    """
    X_list, y_list, g_list = [], [], []

    for subj_idx, (signal, label, _sid) in enumerate(subjects):
        filtered = bandpass_notch(signal)
        epochs   = epoch(filtered)          # (n_e, EPOCH_LEN, N_CHANNELS)
        n_e = epochs.shape[0]
        X_list.append(epochs)
        y_list.append(np.full(n_e, label, dtype=np.int32))
        g_list.append(np.full(n_e, subj_idx, dtype=np.int32))

    X      = np.concatenate(X_list, axis=0).astype(np.float32)
    y      = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    return X, y, groups


def zscore_train_only(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std from X_train over axes (0, 1) — i.e., per channel.
    Apply to both splits.  Never touches X_test statistics.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std

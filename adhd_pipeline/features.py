"""Band-power feature extraction for classical ML baselines."""

import numpy as np
from scipy.signal import welch

from . import config

# np.trapz removed in NumPy 2.0; np.trapezoid added in 2.0
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

BAND_NAMES = list(config.BANDS.keys())   # fixed order for column mapping
N_BANDS    = len(BAND_NAMES)


def _relative_bandpower(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Given a PSD (n_freqs,) and matching frequency axis, return relative power
    in each of the five bands (shape: N_BANDS,).
    """
    total = _trapz(psd, freqs)
    if total == 0:
        return np.zeros(N_BANDS, dtype=np.float32)
    powers = np.empty(N_BANDS, dtype=np.float32)
    for i, band in enumerate(BAND_NAMES):
        lo, hi = config.BANDS[band]
        mask = (freqs >= lo) & (freqs <= hi)
        powers[i] = _trapz(psd[mask], freqs[mask]) / total
    return powers


def bandpower_features(X: np.ndarray) -> np.ndarray:
    """
    X: (n_epochs, EPOCH_LEN, N_CHANNELS)
    Returns: (n_epochs, N_CHANNELS * N_BANDS)  — channel-major order.
    Columns map to (ch0_delta, ch0_theta, …, ch0_gamma, ch1_delta, …).
    """
    n_epochs, epoch_len, n_ch = X.shape
    nperseg = min(epoch_len, 256)
    n_feat  = n_ch * N_BANDS
    out     = np.empty((n_epochs, n_feat), dtype=np.float32)

    for i in range(n_epochs):
        feat_row = []
        for ch in range(n_ch):
            freqs, psd = welch(X[i, :, ch], fs=config.FS, nperseg=nperseg)
            feat_row.append(_relative_bandpower(psd, freqs))
        out[i] = np.concatenate(feat_row)

    return out

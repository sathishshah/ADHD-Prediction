"""SHAP attribution for CNN-BiLSTM under the subject-wise protocol."""

import warnings
from pathlib import Path

import numpy as np

from . import config
from .preprocessing import zscore_train_only
from .models_deep import make_cnn_bilstm

BAND_NAMES    = list(config.BANDS.keys())
CHANNEL_ORDER = config.CHANNEL_ORDER


def compute_shap(
    X: np.ndarray,
    y: np.ndarray,
    splits,
    out_dir: str,
    n_background: int = 100,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train CNN-BiLSTM on fold 0 of *splits*, then compute GradientExplainer SHAP values.

    Returns:
        channel_importance  (N_CHANNELS,)  — mean |SHAP| per channel
        band_importance     (N_BANDS,)     — approximated via bandpass-filter re-explanation
    """
    import shap
    import tensorflow as tf
    from scipy.signal import butter, filtfilt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = splits[0]
    X_tr, X_te = zscore_train_only(X[train_idx], X[test_idx])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.DL_PATIENCE, restore_best_weights=True,
    )

    print("Training CNN-BiLSTM on fold 0 for SHAP …")
    model = make_cnn_bilstm()
    model.fit(
        X_tr, y[train_idx],
        epochs=config.DL_EPOCHS,
        batch_size=config.DL_BATCH,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1,
    )

    rng = np.random.default_rng(config.RANDOM_SEED)

    bg_idx   = rng.choice(len(X_tr), size=min(n_background, len(X_tr)), replace=False)
    samp_idx = rng.choice(len(X_te), size=min(n_samples,    len(X_te)), replace=False)

    background = X_tr[bg_idx]
    samples    = X_te[samp_idx]

    print(f"Running GradientExplainer on {len(samples)} samples …")
    explainer   = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(samples)

    # shap_values: list of arrays or single array depending on shap version
    if isinstance(shap_values, list):
        sv = shap_values[0]          # output neuron 0 (sigmoid)
    else:
        sv = shap_values

    # sv shape: (n_samples, EPOCH_LEN, N_CHANNELS)
    abs_sv = np.abs(sv)

    channel_importance = abs_sv.mean(axis=(0, 1))   # (N_CHANNELS,)

    # Band importance via bandpass-filtered re-explanation
    band_importance = _band_importance_via_filter(
        model, explainer, samples, background, rng
    )

    # ── Save outputs ────────────────────────────────────────────────────────
    np.save(out_dir / "shap_channel_importance.npy", channel_importance)
    np.save(out_dir / "shap_band_importance.npy",    band_importance)

    import pandas as pd
    df_ch = pd.DataFrame({
        "channel_name":  CHANNEL_ORDER,
        "mean_abs_shap": channel_importance,
    }).sort_values("mean_abs_shap", ascending=False)
    df_ch.to_csv(out_dir / "figures" / "shap_channels.csv", index=False)

    print("Top-5 channels by |SHAP|:")
    print(df_ch.head(5).to_string(index=False))

    tf.keras.backend.clear_session()
    return channel_importance, band_importance


def _band_importance_via_filter(
    model, explainer, samples: np.ndarray, background: np.ndarray, rng
) -> np.ndarray:
    """
    For each frequency band, band-pass filter the samples and re-run SHAP.
    Returns (N_BANDS,) array of mean |SHAP| per band.
    """
    from scipy.signal import butter, filtfilt

    nyq = config.FS / 2.0
    band_importance = np.zeros(len(BAND_NAMES), dtype=np.float32)

    for i, band in enumerate(BAND_NAMES):
        lo, hi = config.BANDS[band]
        lo_n, hi_n = lo / nyq, hi / nyq

        # Clamp to valid range for butter
        lo_n = max(lo_n, 1e-4)
        hi_n = min(hi_n, 0.999)

        try:
            b, a = butter(4, [lo_n, hi_n], btype="band")
        except Exception:
            warnings.warn(f"Could not design filter for band {band}; skipping.")
            continue

        filtered = np.empty_like(samples)
        for s_idx in range(samples.shape[0]):
            for ch in range(samples.shape[2]):
                filtered[s_idx, :, ch] = filtfilt(b, a, samples[s_idx, :, ch])

        try:
            sv_band = explainer.shap_values(filtered)
            if isinstance(sv_band, list):
                sv_band = sv_band[0]
            band_importance[i] = np.abs(sv_band).mean()
        except Exception as exc:
            warnings.warn(f"SHAP re-explanation failed for band {band}: {exc}")

    return band_importance

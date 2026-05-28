FS          = 128
BANDPASS    = (0.5, 50.0)
NOTCH_FREQ  = 50.0
EPOCH_SEC   = 4
EPOCH_LEN   = EPOCH_SEC * FS              # 512 samples
N_CHANNELS  = 19
CHANNEL_ORDER = [
    "Fz", "Cz", "Pz", "C3", "T3", "C4", "T4",
    "Fp1", "Fp2", "F3", "F4", "F7", "F8",
    "P3", "P4", "T5", "T6", "O1", "O2",
]
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 50),
}
K_FOLDS     = 10
DL_EPOCHS   = 60
DL_BATCH    = 64
DL_LR       = 1e-3
DL_PATIENCE = 10
DL_DROPOUT  = 0.3
RANDOM_SEED = 42

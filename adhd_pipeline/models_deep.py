"""Deep learning models: CNN-only, LSTM-only, CNN-BiLSTM."""

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config


def _conv_block(x, filters: int, kernel: int) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(config.DL_DROPOUT)(x)
    return x


def make_cnn_bilstm(input_shape: tuple = (config.EPOCH_LEN, config.N_CHANNELS)) -> models.Model:
    """
    CNN-BiLSTM as specified in Methods §3.4.
    Input: (EPOCH_LEN=512, N_CHANNELS=19)
    """
    inp = layers.Input(shape=input_shape)

    x = _conv_block(inp, 64,  kernel=7)
    x = _conv_block(x,   128, kernel=5)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(config.DL_DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name="CNN_BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.DL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_cnn_only(input_shape: tuple = (config.EPOCH_LEN, config.N_CHANNELS)) -> models.Model:
    """CNN-only baseline: same conv stack + GlobalAveragePooling + Dense head."""
    inp = layers.Input(shape=input_shape)

    x = _conv_block(inp, 64,  kernel=7)
    x = _conv_block(x,   128, kernel=5)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(config.DL_DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name="CNN_only")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.DL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_lstm_only(input_shape: tuple = (config.EPOCH_LEN, config.N_CHANNELS)) -> models.Model:
    """LSTM-only baseline: same recurrent stack + Dense head."""
    inp = layers.Input(shape=input_shape)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(32))(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(config.DL_DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name="LSTM_only")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.DL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


DEEP_MODELS = {
    "CNN_only":    make_cnn_only,
    "LSTM_only":   make_lstm_only,
    "CNN_BiLSTM":  make_cnn_bilstm,
}

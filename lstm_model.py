"""
lstm_model.py
Defines the LSTM architecture used for hourly sepsis risk prediction.
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def elementwise_bce(y_true, y_pred):
    """
    Binary cross-entropy WITHOUT averaging over the timestep axis,
    so shape stays (batch, seq_len) and lines up with our
    per-timestep sample_weight array (mask * class weight).
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_true = tf.cast(y_true, y_pred.dtype)
    return -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))


def build_lstm_model(seq_len, n_features, pad_value=-100.0):
    inputs = keras.Input(shape=(seq_len, n_features), name="vitals_sequence")

    x = layers.Masking(mask_value=pad_value)(inputs)

    x = layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)(x)
    x = layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)(x)

    x = layers.TimeDistributed(layers.Dense(16, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
    outputs = layers.Reshape((seq_len,), name="risk_per_hour")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="sepsis_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=elementwise_bce,
        weighted_metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model

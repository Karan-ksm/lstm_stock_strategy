"""LSTM model definition.

Builds a Keras Sequential model from the ModelConfig dataclass. The
architecture is intentionally minimal: an input layer, a stack of N
LSTM layers (each with its own dropout), a small dense projection,
then a single output unit shaped by the configured activation.

For the directional classification setup we use today, the output
activation is sigmoid and the loss is binary_crossentropy. A future
regression variant only needs the output_activation and loss fields
in config.yaml to change; the rest of this file does not move.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import ModelConfig


def build_model(
    model_config: ModelConfig,
    lookback: int,
    n_features: int,
) -> tf.keras.Model:
    """Build and compile a Keras LSTM model.

    Args:
        model_config: Architecture and compile settings from config.yaml.
        lookback: Number of timesteps per input window.
        n_features: Number of feature columns per timestep.

    Returns:
        A compiled Keras model ready for `fit()`.

    Raises:
        ValueError: If the layer stack is empty, or if the last LSTM
            layer is set to return sequences (which would break the
            dense head shape).
    """
    if not model_config.layers:
        raise ValueError("ModelConfig requires at least one LSTM layer")
    if model_config.layers[-1].return_sequences:
        raise ValueError(
            "Last LSTM layer must have return_sequences=false so the "
            "dense head sees a single vector per sample."
        )

    model = models.Sequential()
    model.add(layers.Input(shape=(lookback, n_features)))
    for layer_cfg in model_config.layers:
        model.add(
            layers.LSTM(
                units=layer_cfg.units,
                dropout=layer_cfg.dropout,
                return_sequences=layer_cfg.return_sequences,
            )
        )
    model.add(layers.Dense(model_config.dense_units, activation="relu"))
    model.add(layers.Dense(1, activation=model_config.output_activation))

    model.compile(
        optimizer=_build_optimizer(
            model_config.optimizer, model_config.learning_rate
        ),
        loss=model_config.loss,
        metrics=_default_metrics(model_config),
    )
    return model


def _build_optimizer(
    name: str, learning_rate: float
) -> tf.keras.optimizers.Optimizer:
    """Map an optimizer name from config to a Keras optimizer instance."""
    name_lower = name.lower()
    if name_lower == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name_lower == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name_lower == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name}")


def _default_metrics(model_config: ModelConfig) -> list[str]:
    """Pick reasonable Keras metrics based on the configured output."""
    if model_config.output_activation == "sigmoid":
        return ["accuracy", "AUC"]
    return ["mae"]

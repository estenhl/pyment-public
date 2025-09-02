from typing import Callable

import tensorflow as tf


def loss_factory(name: str) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if name.lower() == 'mse':
        return tf.keras.losses.MeanSquaredError

    raise KeyError(f'Unknown loss {name}')
import tensorflow as tf

from tensorflow.keras.layers import Add, ReLU


def restrict_range(x: tf.Tensor, lower: int, upper: int,
                   name: str = 'restrict_range') -> tf.Tensor:
    assert upper > lower, 'upper must be greater than lower'

    x = ReLU(max_value=upper - lower, name=f'{name}/relu')(x)
    x = Add(name=f'{name}/add')([x, tf.constant([[lower]], dtype=tf.float32)])

    return x

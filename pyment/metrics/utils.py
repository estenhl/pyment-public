import tensorflow as tf


def binarize(x: tf.Tensor, index: int) -> tf.Tensor:
    x = tf.argmax(x, axis=-1)
    x = tf.equal(x, index)
    x = tf.cast(x, tf.int32)

    return x

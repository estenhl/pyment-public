from typing import Callable

import tensorflow as tf 


def metric_factory(name: str) -> tf.keras.metrics.Metric:
    if name.lower() == 'mae':
        return tf.keras.metrics.MeanAbsoluteError()
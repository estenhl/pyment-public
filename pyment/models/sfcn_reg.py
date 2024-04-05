import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Add, Dense, ReLU
from typing import Tuple

from .sfcn import SFCN


class RegressionSFCN(SFCN):
    @classmethod
    def prediction_head(cls, inputs: tf.Tensor, *,
                        prediction_range: Tuple[int] = None, name: str):
        head = Dense(1, activation=None, name=f'{name}_predictions')(inputs)

        if prediction_range is not None:
            lower = np.amin(prediction_range)
            upper = np.amax(prediction_range)

            head = ReLU(max_value=upper - lower,
                        name=f'{name}_restrict_relu')(head)
            head = Add(name=f'{name}_restrict_add')(
                [head, tf.constant([[lower]], dtype=tf.float32)]
            )

        return head

    def __init__(self, *args,
                 prediction_range: Tuple[int] = (3, 95),
                 include_top: bool = True,
                 name: str = 'sfcn-reg',
                 **kwargs):
        self.prediction_range = prediction_range

        super().__init__(*args, prediction_range=prediction_range,
                         include_top=include_top, name=name, **kwargs)


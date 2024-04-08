import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from typing import Tuple

from .sfcn import SFCN


class RankingSFCN(SFCN):
    """ An SFCN-variant for regression through ranking. See
    https://doi.org/10.1109/CVPR.2017.86 for further details.
    """

    @classmethod
    def prediction_head(cls, inputs: tf.Tensor, *,
                        prediction_range: Tuple[int], name: str):
        lower = np.amin(prediction_range)
        upper = np.amax(prediction_range)

        head = Dense(upper - lower - 1, activation='sigmoid',
                     name=f'{name}_predictions')(inputs)

        return head

    def __init__(self, *args,
                 prediction_range: Tuple[int] = (3, 95),
                 include_top: bool = True,
                 name: str = 'sfcn-rank',
                 **kwargs):
        self.prediction_range = prediction_range

        super().__init__(*args, prediction_range=prediction_range,
                         include_top=include_top, name=name, **kwargs)


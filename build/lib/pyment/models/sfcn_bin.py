""" Module containing the binary SFCN implementation. """
import tensorflow as tf

from tensorflow.keras.layers import Dense

from .sfcn import SFCN


class BinarySFCN(SFCN):
    """ An SFCN-variant for binary classification. """

    @classmethod
    def prediction_head(cls, inputs: tf.Tensor, *, name: str) -> tf.Tensor:
        head = Dense(1, activation='sigmoid',
                     name=f'{name}_predictions')(inputs)

        return head

    def __init__(self, *args,
                 include_top: bool = True,
                 name: str = 'sfcn-bin',
                 **kwargs):
        super().__init__(*args, include_top=include_top, name=name, **kwargs)

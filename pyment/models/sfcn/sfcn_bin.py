"""SFCN with a single sigmoid output for binary classification."""

from tensorflow import Tensor
from tensorflow.keras.layers import Dense

from .sfcn import SFCN


class BinarySFCN(SFCN):
    """SFCN for binary classification tasks.

    The prediction head is a single Dense unit with sigmoid activation,
    producing a probability in [0, 1].
    """

    @staticmethod
    def construct_prediction_head(
        bottleneck: Tensor,
        name: str,
    ) -> Tensor:
        layer = Dense(1, activation='sigmoid', name=f'{name}_predictions')

        return layer(bottleneck)

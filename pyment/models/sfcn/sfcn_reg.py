"""SFCN with a single linear output for regression."""

from tensorflow import Tensor
from tensorflow.keras.layers import Dense

from .sfcn import SFCN


class RegressionSFCN(SFCN):
    """SFCN for continuous regression tasks.

    The prediction head is a single Dense unit with linear activation.
    """

    @staticmethod
    def construct_prediction_head(
        bottleneck: Tensor,
        name: str,
    ) -> Tensor:
        layer = Dense(1, activation=None, name=f'{name}/predictions')

        return layer(bottleneck)

from tensorflow import Tensor
from tensorflow.keras.layers import Dense

from .sfcn import SFCN


class RegressionSFCN(SFCN):
    @classmethod
    def construct_prediction_head(
        cls, 
        bottleneck: Tensor, 
        name: str
    ) -> Tensor:
        layer = Dense(1, activation=None, name=f'{name}/predictions')

        return layer(bottleneck)

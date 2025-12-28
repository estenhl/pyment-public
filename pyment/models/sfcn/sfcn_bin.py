from tensorflow import Tensor
from tensorflow.keras.layers import Dense

from .sfcn import SFCN


class BinarySFCN(SFCN):
    @classmethod
    def construct_prediction_head(
        cls,
        bottleneck: Tensor,
        name: str
    ) -> Tensor:
        layer = Dense(1, activation='sigmoid', name=f'{name}/predictions')

        return layer(bottleneck)
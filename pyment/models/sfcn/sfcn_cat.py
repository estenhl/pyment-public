"""SFCN with a softmax output for categorical classification."""

from tensorflow import Tensor
from tensorflow.keras.layers import Dense

from .sfcn import SFCN


class CategoricalSFCN(SFCN):
    """SFCN for multi-class classification tasks.

    The prediction head is a Dense layer with softmax activation,
    producing a probability distribution over ``num_classes`` classes.
    """

    def __init__(self, *, num_classes: int, **kwargs):
        self._num_classes = num_classes
        super().__init__(**kwargs)

    def construct_prediction_head(
        self,
        bottleneck: Tensor,
        name: str,
    ) -> Tensor:
        layer = Dense(
            self._num_classes,
            activation='softmax',
            name=f'{name}_predictions',
        )

        return layer(bottleneck)

import tensorflow as tf

from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, \
                                    Dropout, Input, MaxPooling3D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer
from typing import Callable, Tuple, Union

from .model import Model
from .utils import get_global_pooling_layer


class SFCN(Model):
    FILTERS = [32, 64, 128, 256, 256, 64]

    @classmethod
    def prediction_head(cls, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError('Base SFCN model has no prediction head and '
                                  'should not be initialized with '
                                  'include_top=True')

    def __init__(self, *,
                 input_shape: Tuple[int] = (167, 212, 160, 1),
                 pooling: Union[str, Callable] = 'avg',
                 include_top: bool = False,
                 dropout: float = 0.0,
                 regularizer: Regularizer = None,
                 activation: Union[str, Callable] = 'relu',
                 weights: str = None,
                 name: str = 'sfcn',
                 **kwargs):

        self.input_layer = Input(input_shape, name=f'{name}_inputs')
        curr = self.input_layer

        # Adds channel-dimension if necessary
        if len(input_shape) == 3:
            curr = Reshape(input_shape + (1,),
                           name=f'{name}_expand-dims')(curr)

        for i in range(len(self.FILTERS) - 1):
            block_name = f'{name}_block-{i}'

            curr = Conv3D(filters=self.FILTERS[i],
                          kernel_size=(3, 3, 3),
                          padding='SAME',
                          activation=None,
                          kernel_regularizer=regularizer,
                          name=f'{block_name}_conv')(curr)
            curr = BatchNormalization(name=f'{block_name}_norm')(curr)
            curr = Activation(activation,
                              name=f'{block_name}_{activation}')(curr)
            curr = MaxPooling3D(pool_size=(2, 2, 2),
                                name=f'{block_name}_pool')(curr)

        curr = Conv3D(filters=self.FILTERS[-1],
                      kernel_size=(1, 1, 1),
                      padding='SAME',
                      activation=None,
                      name=f'{name}_top_conv')(curr)
        curr = BatchNormalization(name=f'{name}_top_norm')(curr)
        curr = Activation(activation, name=f'{name}_top_{activation}')(curr)

        pooling = get_global_pooling_layer(pooling, dimensions=3)
        curr = pooling(name=f'{name}_top_pool')(curr)

        self.bottleneck = curr

        if include_top:
            curr = Dropout(dropout, name=f'{name}_top_dropout')(curr)
            curr = self.prediction_head(curr, name=name, **kwargs)

        super().__init__(self.input_layer, curr, include_top=include_top,
                         weights=weights, name=name)


import tensorflow as tf

from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, \
                                    Dense, Dropout, GlobalAveragePooling3D, \
                                    Input, MaxPooling3D, Reshape
from tensorflow.keras.regularizers import l2
from typing import List, Tuple

from .model import Model
from .model_type import ModelType
from .utils import restrict_range


class RegressionSFCN(Model):
    @property
    def type(self) -> ModelType:
        return ModelType.REGRESSION

    def __init__(self, *, input_shape: Tuple[int, int, int] = (167, 212, 160),
                 dropout: float = .0, weight_decay: float = .0,
                 activation: str = 'relu', include_top: bool = True,
                 depths: List[int] = [32, 64, 128, 256, 256, 64],
                 prediction_range: Tuple[float, float] = (3, 95),
                 name: str = 'Regression3DSFCN', weights: str = None,
                 domains: int = None):
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)

        if domains is not None:
            from adabn import AdaptiveBatchNormalization

        regularizer = l2(weight_decay) if weight_decay is not None else None

        inputs = Input(input_shape, name=f'{name}/inputs')

        if domains is not None:
            num_domains = domains
            domains = Input((), name=f'{name}/domains', dtype=tf.int32)

        x = inputs
        x = Reshape(input_shape + (1,), name=f'{name}/expand_dims')(x)

        for i in range(5):
            x = Conv3D(depths[i], (3, 3, 3), padding='SAME',
                       activation=None, kernel_regularizer=regularizer,
                       bias_regularizer=regularizer,
                       name=f'{name}/block{i+1}/conv')(x)

            if domains is None:
                x = BatchNormalization(name=f'{name}/block{i+1}/norm')(x)
            else:
                x = AdaptiveBatchNormalization(
                    domains=num_domains,
                    beta_regularizer=regularizer,
                    gamma_regularizer=regularizer
                )([x, domains])

            x = Activation(activation,
                           name=f'{name}/block{i+1}/{activation}')(x)
            x = MaxPooling3D((2, 2, 2), name=f'{name}/block{i+1}/pool')(x)

        x = Conv3D(depths[-1], (1, 1, 1), padding='SAME', activation=None,
                   kernel_regularizer=regularizer,
                   bias_regularizer=regularizer, name=f'{name}/top/conv')(x)

        if domains is None:
            x = BatchNormalization(name=f'{name}/top/norm')(x)
        else:
            x = AdaptiveBatchNormalization(
                domains=num_domains,
                beta_regularizer=regularizer,
                gamma_regularizer=regularizer
            )([x, domains])

        x = Activation(activation, name=f'{name}/top/{activation}')(x)
        x = GlobalAveragePooling3D(name=f'{name}/top/pool')(x)
        bottleneck = x

        x = Dropout(dropout, name=f'{name}/top/dropout')(x)
        x = Dense(1, activation=None, name=f'{name}/predictions')(x)

        if prediction_range is not None:
            x = restrict_range(x, *prediction_range, name=f'{name}/restrict')

        if not include_top:
            x = bottleneck

        if domains is not None:
            inputs = [inputs, domains]

        super().__init__(inputs, x, weights=weights, include_top=include_top,
                         name=name)

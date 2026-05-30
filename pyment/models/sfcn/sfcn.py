from abc import abstractmethod
from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv3D,
    Dropout,
    GlobalAveragePooling3D,
    GlobalMaxPooling3D,
    Input,
    MaxPooling3D,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer

from ..utils.ensure_weights import ensure_weights


class SFCN(Model):
    @staticmethod
    @abstractmethod
    def construct_prediction_head(
        bottleneck: tf.Tensor,
        name: str,
    ) -> tf.Tensor:
        pass

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int] = (224, 192, 224),
        include_top: bool = True,
        pooling: str = 'avg',
        dropout: float = 0.0,
        regularizer: Regularizer | None = None,
        activation: Any = 'relu',
        name: str = 'SFCN',
        weights: str | None = None,
    ):
        inputs = Input(input_shape, name=f'{name}_inputs')
        x = Reshape(input_shape + (1,), name=f'{name}_expand-dims')(inputs)

        NUM_FILTERS = [32, 64, 128, 256, 256, 64]

        for i, num_filters in enumerate(NUM_FILTERS[:-1]):
            x = Conv3D(
                num_filters,
                (3, 3, 3),
                padding='SAME',
                activation=None,
                kernel_regularizer=regularizer,
                name=f'{name}_block-{i}_conv',
            )(x)
            x = BatchNormalization(name=f'{name}_block-{i}_norm')(x)
            x = Activation(activation, name=f'{name}_block-{i}_activation')(x)
            x = MaxPooling3D((2, 2, 2), name=f'{name}_block-{i}_pool')(x)

        x = Conv3D(
            NUM_FILTERS[-1],
            (1, 1, 1),
            padding='SAME',
            activation=None,
            name=f'{name}_top_conv',
        )(x)
        x = BatchNormalization(name=f'{name}_top_norm')(x)
        x = Activation(activation, name=f'{name}_top_activation')(x)

        POOLING_OPTIONS = {
            'avg': GlobalAveragePooling3D,
            'max': GlobalMaxPooling3D,
        }
        pooling_cls = POOLING_OPTIONS.get(pooling)

        if pooling_cls is None:
            raise ValueError(f'Unknown pooling layer {pooling}')

        x = pooling_cls(name=f'{name}_top_pool')(x)

        self.bottleneck = x

        if include_top:
            x = Dropout(dropout, name=f'{name}_dropout')(x)
            x = self.construct_prediction_head(x, name=f'{name}_predictions')

        super().__init__(inputs, x)

        if weights:
            weights = ensure_weights(weights)
            self.load_weights(weights)

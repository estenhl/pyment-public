from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, \
                                    Dense, Dropout, Flatten, Input, \
                                    MaxPooling3D, Reshape
from tensorflow.keras.regularizers import l2
from typing import List, Tuple

from .model import Model
from .model_type import ModelType
from .utils import restrict_range


class RegressionVGG11(Model):
    @property
    def type(self) -> ModelType:
        return ModelType.REGRESSION

    def __init__(self, *,  input_shape: Tuple[int, int, int] = (167, 212, 160),
                 dropout: float = .0, weight_decay: float = .0,
                 activation: str = 'relu', include_top: bool = True,
                 depths: List[int] = [32, 32, 64, 64, 64, 64, 128, 128,
                                      256, 256, 64],
                 prediction_range: Tuple[float, float] = (3, 95),
                 name: str = 'Regression3DVGG19', weights: str = None):
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)

        regularizer = l2(weight_decay) if weight_decay is not None else None

        inputs = Input(input_shape, name=f'{name}/inputs')

        x = inputs
        x = Reshape(input_shape + (1,), name=f'{name}/expand_dims')(x)

        for i in range(2):
            x = Conv3D(depths[i], (3, 3, 3), padding='SAME',
                        activation=None, kernel_regularizer=regularizer,
                        bias_regularizer=regularizer,
                        name=f'{name}/block{i+1}/conv')(x)
            x = BatchNormalization(name=f'{name}/block{i+1}/norm')(x)
            x = Activation(activation,
                           name=f'{name}/block{i+1}/{activation}')(x)
            x = MaxPooling3D((2, 2, 2), name=f'{name}/block{i+1}/pool')(x)

        for i in range(2, 5):
            x = Conv3D(depths[i], (3, 3, 3), padding='SAME',
                        activation=None, kernel_regularizer=regularizer,
                        bias_regularizer=regularizer,
                        name=f'{name}/block{i+1}/a/conv')(x)
            x = BatchNormalization(name=f'{name}/block{i+1}/a/norm')(x)
            x = Activation(activation,
                           name=f'{name}/block{i+1}/a/{activation}')(x)
            x = Conv3D(depths[i+1], (3, 3, 3), padding='SAME',
                        activation=None, kernel_regularizer=regularizer,
                        bias_regularizer=regularizer,
                        name=f'{name}/block{i+1}/b/conv')(x)
            x = BatchNormalization(name=f'{name}/block{i+1}/b/norm')(x)
            x = Activation(activation,
                           name=f'{name}/block{i+1}/b/{activation}')(x)
            x = MaxPooling3D((2, 2, 2), name=f'{name}/block{i+1}/pool')(x)

        x = Flatten(name=f'{name}/flatten')(x)
        x = Dropout(dropout)(x)
        x = Dense(depths[8], activation=None, name=f'{name}/fc1')(x)
        x = BatchNormalization(name=f'{name}/fc1/norm')(x)
        x = Activation(activation, name=f'{name}/fc1/{activation}')(x)
        x = Dense(depths[9], activation=None, name=f'{name}/fc2')(x)
        x = BatchNormalization(name=f'{name}/fc2/norm')(x)
        x = Activation(activation, name=f'{name}/fc2/{activation}')(x)
        x = Dense(depths[10], activation=None, name=f'{name}/fc3')(x)
        x = BatchNormalization(name=f'{name}/fc3/norm')(x)
        x = Activation(activation, name=f'{name}/fc3/{activation}')(x)
        bottleneck = x

        x = Dense(1, activation=None, name=f'{name}/predictions')(x)

        if prediction_range is not None:
            x = restrict_range(x, *prediction_range, name=f'{name}/restrict')

        if not include_top:
            x = bottleneck

        super().__init__(inputs, x, weights=weights, include_top=include_top,
                         name=name)

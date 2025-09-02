from tensorflow import concat, Tensor
from tensorflow.keras.layers import Activation, BatchNormalization, Dense

from .sfcn import SFCN


class MultiTaskSFCN(SFCN):
    @classmethod
    def construct_prediction_head(
        cls, 
        bottleneck: Tensor, 
        name: str
    ) -> Tensor:
        x = bottleneck
        depths = []

        for i in range(len(depths)):
            x = Dense(depths[i], activation=None,
                      name=f'{name}_dense-{i+1}_dense')(x)
            x = BatchNormalization(name=f'{name}_dense-{i+1}_norm')(x)
            x = Activation('relu', name=f'{name}_dense-{i+1}_activation')(x)

        heads = [
            Dense(1, activation=None, name=f'{name}_predictions_age'),
            Dense(1, activation='sigmoid', name=f'{name}_predictions_sex'),
            Dense(1, activation='sigmoid',
                  name=f'{name}_predictions_handedness'),
            Dense(1, activation=None, name=f'{name}_predictions_bmi'),
            Dense(1, activation=None,
                  name=f'{name}_predictions_fluid_intelligence'),
            Dense(1, activation=None, name=f'{name}_predictions_neuroticism')
        ]
        heads = [head(x) for head in heads]

        return concat(heads, axis=-1)

    def __init__(self, *args,
        pooling: str = 'max',
        name: str = 'SFCNMulti',
        **kwargs
    ):
        super().__init__(*args, pooling=pooling, name=name, **kwargs)

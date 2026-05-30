"""SFCN with six parallel outputs for multi-task neuroimaging
prediction."""

import logging

from tensorflow import Tensor
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv3D,
    Dense,
)

from .sfcn import SFCN

logger = logging.getLogger(__name__)


class MultiTaskSFCN(SFCN):
    """SFCN for simultaneous prediction of six neuroimaging targets.

    The prediction head fans out to six independent Dense units whose
    outputs are concatenated along the last axis. Output order:
    age, sex, handedness, bmi, fluid_intelligence, neuroticism.
    Sex and handedness use sigmoid activation; the remaining four are
    linear.
    """

    @staticmethod
    def construct_prediction_head(
        bottleneck: Tensor,
        name: str,
    ) -> Tensor:
        x = bottleneck
        depths = []

        for i in range(len(depths)):
            x = Dense(
                depths[i], activation=None, name=f'{name}_dense-{i + 1}_dense'
            )(x)
            x = BatchNormalization(name=f'{name}_dense-{i + 1}_norm')(x)
            x = Activation('relu', name=f'{name}_dense-{i + 1}_activation')(x)

        heads = [
            Dense(1, activation=None, name=f'{name}_predictions_age'),
            Dense(1, activation='sigmoid', name=f'{name}_predictions_sex'),
            Dense(
                1, activation='sigmoid', name=f'{name}_predictions_handedness'
            ),
            Dense(1, activation=None, name=f'{name}_predictions_bmi'),
            Dense(
                1,
                activation=None,
                name=f'{name}_predictions_fluid_intelligence',
            ),
            Dense(1, activation=None, name=f'{name}_predictions_neuroticism'),
        ]
        heads = [head(x) for head in heads]

        return Concatenate(axis=-1)(heads)

    TARGETS = [
        'age',
        'sex',
        'handedness',
        'bmi',
        'fluid_intelligence',
        'neuroticism',
    ]

    def __init__(
        self, *args, pooling: str = 'max', name: str = 'SFCNMulti', **kwargs
    ):
        super().__init__(*args, pooling=pooling, name=name, **kwargs)

    def transfer_weights_to_single_task_model(
        self,
        model: SFCN,
        target: str | None = None,
    ) -> None:
        """Transfer backbone weights (and optionally a task head) into
        a single-task SFCN-model.

        Parameters
        ----------
        model:
            Target single-task SFCN to receive the weights.
        target:
            One of the six pretrained task names. When given, the
            corresponding head weights are also transferred into the
            target model's prediction layer.
        """
        src_conv = [layer for layer in self.layers if isinstance(layer, Conv3D)]
        dst_conv = [
            layer for layer in model.layers if isinstance(layer, Conv3D)
        ]
        for src, dst in zip(src_conv, dst_conv):
            dst.set_weights(src.get_weights())

        src_norm = [
            layer
            for layer in self.layers
            if isinstance(layer, BatchNormalization)
        ]
        dst_norm = [
            layer
            for layer in model.layers
            if isinstance(layer, BatchNormalization)
        ]
        for src, dst in zip(src_norm, dst_norm):
            dst.set_weights(src.get_weights())

        if target is None:
            return

        if target not in self.TARGETS:
            logger.warning(
                'Unknown target %s, not transferring prediction head weights',
                target,
            )
            return

        head_layers = [
            layer for layer in self.layers if isinstance(layer, Dense)
        ]
        src_head = head_layers[self.TARGETS.index(target)]
        dst_head = next(
            layer for layer in model.layers if isinstance(layer, Dense)
        )
        dst_head.set_weights(src_head.get_weights())

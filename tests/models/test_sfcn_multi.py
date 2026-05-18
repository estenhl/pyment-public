import logging

import numpy as np
import pytest
from tensorflow.keras.layers import Conv3D

from pyment.models.sfcn import MultiTaskSFCN, RegressionSFCN

INPUT_SHAPE = (32, 32, 32)


@pytest.fixture(scope='module')
def backbone():
    return MultiTaskSFCN(input_shape=INPUT_SHAPE)


@pytest.fixture(scope='module')
def regression_model():
    return RegressionSFCN(input_shape=INPUT_SHAPE)


def test_backbone_conv_weights_are_transferred(backbone, regression_model):
    backbone.transfer_weights_to_single_task_model(regression_model)

    src_conv = [layer for layer in backbone.layers if isinstance(layer, Conv3D)]
    dst_conv = [
        layer for layer in regression_model.layers if isinstance(layer, Conv3D)
    ]

    for src, dst in zip(src_conv, dst_conv):
        for sw, dw in zip(src.get_weights(), dst.get_weights()):
            assert np.array_equal(sw, dw), (
                f'Expected Conv3D layer weights to match after transfer '
                f'(layer {src.name} → {dst.name})'
            )


def test_backbone_norm_weights_are_transferred(backbone, regression_model):
    backbone.transfer_weights_to_single_task_model(regression_model)

    from tensorflow.keras.layers import BatchNormalization

    src_norm = [
        layer
        for layer in backbone.layers
        if isinstance(layer, BatchNormalization)
    ]
    dst_norm = [
        layer
        for layer in regression_model.layers
        if isinstance(layer, BatchNormalization)
    ]

    for src, dst in zip(src_norm, dst_norm):
        for sw, dw in zip(src.get_weights(), dst.get_weights()):
            assert np.array_equal(sw, dw), (
                f'Expected BatchNormalization layer weights to match after '
                f'transfer (layer {src.name} → {dst.name})'
            )


def test_head_is_not_transferred_without_target(backbone, regression_model):
    from tensorflow.keras.layers import Dense

    dst_head = next(
        layer for layer in regression_model.layers if isinstance(layer, Dense)
    )
    weights_before = [w.copy() for w in dst_head.get_weights()]

    backbone.transfer_weights_to_single_task_model(regression_model)

    for before, after in zip(weights_before, dst_head.get_weights()):
        assert np.array_equal(before, after), (
            'Expected prediction head weights to be unchanged when no '
            'target is given'
        )


@pytest.mark.parametrize('target', MultiTaskSFCN.TARGETS)
def test_head_weights_are_transferred_for_known_target(target):
    from tensorflow.keras.layers import Dense

    backbone = MultiTaskSFCN(input_shape=INPUT_SHAPE)
    model = RegressionSFCN(input_shape=INPUT_SHAPE)

    backbone.transfer_weights_to_single_task_model(model, target=target)

    src_head = [layer for layer in backbone.layers if isinstance(layer, Dense)][
        MultiTaskSFCN.TARGETS.index(target)
    ]
    dst_head = next(layer for layer in model.layers if isinstance(layer, Dense))

    for sw, dw in zip(src_head.get_weights(), dst_head.get_weights()):
        assert np.array_equal(sw, dw), (
            f'Expected prediction head weights to match after transfer '
            f'for target {target!r}'
        )


def test_unknown_target_logs_warning_and_skips_head(
    backbone, regression_model, caplog
):
    from tensorflow.keras.layers import Dense

    dst_head = next(
        layer for layer in regression_model.layers if isinstance(layer, Dense)
    )
    weights_before = [w.copy() for w in dst_head.get_weights()]

    with caplog.at_level(logging.WARNING):
        backbone.transfer_weights_to_single_task_model(
            regression_model, target='unknown'
        )

    assert 'unknown' in caplog.text, (
        'Expected a warning containing the unrecognised target name'
    )
    for before, after in zip(weights_before, dst_head.get_weights()):
        assert np.array_equal(before, after), (
            'Expected prediction head weights to be unchanged for an '
            'unknown target'
        )

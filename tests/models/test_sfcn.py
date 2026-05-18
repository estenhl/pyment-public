import pytest
from tensorflow.keras.layers import (
    GlobalAveragePooling3D,
    GlobalMaxPooling3D,
    MaxPooling3D,
)

from pyment.models.sfcn import BinarySFCN

INPUT_SHAPE = (32, 32, 32)
_POOLING_TYPES = (GlobalAveragePooling3D, GlobalMaxPooling3D, MaxPooling3D)


def test_default_input_shape():
    model = BinarySFCN()
    assert model.input_shape == (None, 224, 192, 224), (
        'Expected SFCN to yield input shape (None, 224, 192, 224) by default'
    )


def test_custom_input_shape():
    model = BinarySFCN(input_shape=INPUT_SHAPE)
    assert model.input_shape == (None,) + INPUT_SHAPE, (
        'Expected SFCN to use input_shape when provided'
    )


def test_output_shape_with_top():
    model = BinarySFCN(input_shape=INPUT_SHAPE, include_top=True)
    assert model.output_shape == (None, 1), (
        'Expected SFCN to yield output shape (None, 1) for include_top=True'
    )


@pytest.mark.parametrize(
    'pooling, expected',
    [
        ('avg', GlobalAveragePooling3D),
        ('max', GlobalMaxPooling3D),
    ],
    ids=['avg', 'max'],
)
def test_pooling(pooling, expected):
    model = BinarySFCN(input_shape=INPUT_SHAPE, pooling=pooling)
    pool_layers = [
        layer for layer in model.layers if isinstance(layer, _POOLING_TYPES)
    ]
    assert isinstance(pool_layers[-1], expected), (
        f'Expected last pooling layer to be {expected.__name__} '
        f'for pooling={pooling!r}'
    )


def test_unknown_pooling_raises():
    with pytest.raises(ValueError, match='Unknown pooling layer'):
        BinarySFCN(input_shape=INPUT_SHAPE, pooling='unknown')

import pytest
import tensorflow as tf

from pyment.models.sfcn import BinarySFCN, MultiTaskSFCN, RegressionSFCN

INPUT_SHAPE = (32, 32, 32)


@pytest.fixture(scope='module')
def binary_sfcn():
    return BinarySFCN(input_shape=INPUT_SHAPE)


@pytest.fixture(scope='module')
def regression_sfcn():
    return RegressionSFCN(input_shape=INPUT_SHAPE)


@pytest.fixture(scope='module')
def multi_sfcn():
    return MultiTaskSFCN(input_shape=INPUT_SHAPE)


def test_binary_sfcn(binary_sfcn):
    assert binary_sfcn.output_shape == (None, 1), (
        'Expected BinarySFCN to yield output shape (None, 1)'
    )
    last = binary_sfcn.layers[-1]
    assert last.activation == tf.keras.activations.sigmoid, (
        'Expected BinarySFCN to use sigmoid activation on the prediction layer'
    )


def test_regression_sfcn(regression_sfcn):
    assert regression_sfcn.output_shape == (None, 1), (
        'Expected RegressionSFCN to yield output shape (None, 1)'
    )
    last = regression_sfcn.layers[-1]
    assert last.activation == tf.keras.activations.linear, (
        'Expected RegressionSFCN to use linear activation '
        'on the prediction layer'
    )


def test_multi_sfcn(multi_sfcn):
    assert multi_sfcn.output_shape == (None, 6), (
        'Expected MultiTaskSFCN to yield output shape (None, 6)'
    )


# Layer counts are pinned to the pretrained checkpoint architecture.
# Any change (added, removed, or reordered layers) will silently
# break weight loading.
@pytest.mark.parametrize(
    'fixture_name, expected_count',
    [
        ('binary_sfcn', 28),
        ('regression_sfcn', 28),
        ('multi_sfcn', 34),
    ],
)
def test_layer_count(fixture_name, expected_count, request):
    model = request.getfixturevalue(fixture_name)
    assert len(model.layers) == expected_count, (
        f'Expected {type(model).__name__} to have {expected_count} layers'
    )

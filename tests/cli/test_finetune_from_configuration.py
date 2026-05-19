import pytest
import tensorflow as tf

from pyment.cli.finetune_from_configuration import _resolve_optimizer


@pytest.mark.parametrize(
    'name, expected_cls',
    [
        ('adam', tf.keras.optimizers.legacy.Adam),
        ('sgd', tf.keras.optimizers.legacy.SGD),
        ('rmsprop', tf.keras.optimizers.legacy.RMSprop),
    ],
)
def test_known_optimizer_returns_instance(name, expected_cls):
    optimizer = _resolve_optimizer(name)
    assert isinstance(optimizer, expected_cls), (
        f'Expected _resolve_optimizer to return an instance of '
        f'{expected_cls.__name__} for {name!r}'
    )


def test_unknown_optimizer_name_raises():
    with pytest.raises(KeyError, match='Unknown optimizer name'):
        _resolve_optimizer('not_a_real_optimizer')


@pytest.mark.parametrize('bad_input', [None, 42, ['adam'], object()])
def test_non_string_input_raises(bad_input):
    with pytest.raises(ValueError, match='must be a string'):
        _resolve_optimizer(bad_input)

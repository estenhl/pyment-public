import pytest

from pyment.configurations.target_configuration import (
    BinaryTargetConfiguration,
    RegressionTargetConfiguration,
    compile_target_encoder,
)


def test_binary_config_returns_encoder():
    config = BinaryTargetConfiguration(name='sex', kind='binary', labels=[0, 1])
    assert compile_target_encoder(config) is not None, (
        'Expected compile_target_encoder to return an encoder for a binary '
        'target'
    )


def test_regression_config_returns_none():
    config = RegressionTargetConfiguration(name='age', kind='regression')
    assert compile_target_encoder(config) is None, (
        'Expected compile_target_encoder to return no encoder for a regression '
        'target'
    )


@pytest.mark.parametrize(
    'label, expected',
    [(0, 0), (1, 1)],
    ids=['first', 'second'],
)
def test_encoder_maps_label_to_index(label, expected):
    config = BinaryTargetConfiguration(name='sex', kind='binary', labels=[0, 1])
    encoder = compile_target_encoder(config)
    assert encoder is not None and encoder(label) == expected, (
        f'Expected binary encoder to return {expected} for label {label} '
        'according to its given configuration'
    )

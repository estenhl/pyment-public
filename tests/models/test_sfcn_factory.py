import pytest

from pyment.models.sfcn import (
    BinarySFCN,
    MultiTaskSFCN,
    RegressionSFCN,
    sfcn_factory,
)


@pytest.mark.parametrize(
    'name, expected',
    [
        ('sfcn-reg', RegressionSFCN),
        ('regression', RegressionSFCN),
        ('sfcn-bin', BinarySFCN),
        ('binary', BinarySFCN),
        ('sfcn-multi', MultiTaskSFCN),
        ('multi', MultiTaskSFCN),
    ],
    ids=[
        'sfcn-reg',
        'regression',
        'sfcn-bin',
        'binary',
        'sfcn-multi',
        'multi',
    ],
)
def test_factory_returns_correct_class(name, expected):
    assert sfcn_factory(name) is expected, (
        f'Expected sfcn_factory to return {expected.__name__} for {name!r}'
    )


def test_factory_raises_for_unknown_type():
    with pytest.raises(ValueError, match='Unknown SFCN type'):
        sfcn_factory('unknown')

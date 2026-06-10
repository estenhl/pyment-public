from .sfcn import SFCN
from .sfcn_bin import BinarySFCN
from .sfcn_cat import CategoricalSFCN
from .sfcn_multi import MultiTaskSFCN
from .sfcn_reg import RegressionSFCN


def sfcn_factory(model_type: str) -> type[SFCN]:
    if model_type in ['sfcn-reg', 'regression']:
        return RegressionSFCN
    elif model_type in ['sfcn-bin', 'binary']:
        return BinarySFCN
    elif model_type in ['sfcn-cat', 'categorical']:
        return CategoricalSFCN
    elif model_type in ['sfcn-multi', 'multi']:
        return MultiTaskSFCN

    raise ValueError(f'Unknown SFCN type {model_type}')


__all__ = [
    'SFCN',
    'BinarySFCN',
    'CategoricalSFCN',
    'MultiTaskSFCN',
    'RegressionSFCN',
    'sfcn_factory',
]

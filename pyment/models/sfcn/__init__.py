from .sfcn import SFCN
from .sfcn_bin import BinarySFCN
from .sfcn_multi import MultiTaskSFCN
from .sfcn_reg import RegressionSFCN


def sfcn_factory(model_type: str):
    if model_type in ['sfcn-reg', 'regression']:
        return RegressionSFCN
    elif model_type in ['sfcn-bin', 'binary']:
        return BinarySFCN
    elif model_type in ['sfcn-multi', 'multi']:
        return MultiTaskSFCN

    raise ValueError(f'Unknown SFCN type {model_type}')

__all__ = [
    'sfcn_factory',
    'BinarySFCN',
    'MultiTaskSFCN',
    'RegressionSFCN',
    'SFCN'
]

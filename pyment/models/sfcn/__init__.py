from .sfcn import SFCN
from .sfcn_multi import MultiTaskSFCN
from .sfcn_reg import RegressionSFCN


def sfcn_factory(model_type: str):
    if model_type in ['sfcn-reg', 'regression']:
        return RegressionSFCN
    elif model_type in ['sfcn-multi', 'multi']:
        return MultiTaskSFCN

    raise ValueError(f'Unknown SFCN type {model_type}')

__all__ = ['sfcn_factory', 'SFCN', 'MultiTaskSFCN', 'RegressionSFCN']
""" A module containing implementations for the various model
architectures. """

from .model import Model
from .sfcn import SFCN
from .sfcn_bin import BinarySFCN
from .sfcn_rank import RankingSFCN
from .sfcn_reg import RegressionSFCN
from .sfcn_sm import SoftClassificationSFCN


def get_model_class(name: str) -> Model:
    if name.lower() in ['sfcn-bin', 'binarysfcn']:
        return BinarySFCN
    elif name.lower() in ['sfcn-rank', 'rankingsfcn']:
        return RankingSFCN
    elif name.lower() in ['sfcn-reg', 'regressionsfcn']:
        return RegressionSFCN
    elif name.lower() in ['sfcn-sm', 'softclassificationsfcn']:
        return SoftClassificationSFCN

    raise ValueError(f'Invalid model name {name}')

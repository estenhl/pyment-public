from .model_type import ModelType
from .regression_sfcn import RegressionSFCN


def get(model_name: str, **kwargs):
    if model_name.lower() in ['regressionsfcn', 'sfcn-reg']:
        return RegressionSFCN(**kwargs)
    else:
        raise ValueError(f'Unknown model {model_name}')
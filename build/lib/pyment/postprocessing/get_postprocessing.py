import numpy as np

from .sfcn_rank import sfcn_rank_brain_age_2022
from .sfcn_sm import sfcn_sm_brain_age_2022


def squeeze(values: np.ndarray) -> np.ndarray:
    return values[...,0]

_postprocessors = {
    ('RankingSFCN', 'brain-age-2022'): sfcn_rank_brain_age_2022,
    ('RegressionSFCN', 'brain-age-2022'): squeeze,
    ('SoftClassificationSFCN', 'brain-age-2022'): sfcn_sm_brain_age_2022,
    ('BinarySFCN', 'dementia-2024'): squeeze,
    ('BinarySFCN', 'dementia-2024-fold-0'): squeeze,
    ('BinarySFCN', 'dementia-2024-fold-1'): squeeze,
    ('BinarySFCN', 'dementia-2024-fold-2'): squeeze,
    ('BinarySFCN', 'dementia-2024-fold-3'): squeeze,
    ('BinarySFCN', 'dementia-2024-fold-4'): squeeze,
}

def get_postprocessing(modelname: str, weights: str):
    key = (modelname, weights)

    if not key in _postprocessors:
        raise ValueError('Invalid model/weights combination for retrieving '
                         f'postprocessing: {(modelname, weights)}')

    postprocessor = _postprocessors[key]

    if postprocessor is None:
        return lambda x: x

    return postprocessor

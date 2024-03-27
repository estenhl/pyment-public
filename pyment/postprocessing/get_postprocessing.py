from .sfcn_rank import sfcn_rank_brain_age_2022
from .sfcn_reg import sfcn_reg_brain_age_2022
from .sfcn_sm import sfcn_sm_brain_age_2022


_postprocessors = {
    ('RankingSFCN', 'brain-age-2022'): sfcn_rank_brain_age_2022,
    ('RegressionSFCN', 'brain-age-2022'): sfcn_reg_brain_age_2022,
    ('SoftClassificationSFCN', 'brain-age-2022'): sfcn_sm_brain_age_2022,
    # Used by docker containers
    ('RankingSFCN', '/code/weights.h5'): sfcn_rank_brain_age_2022,
    ('RegressionSFCN', '/code/weights.h5'): sfcn_reg_brain_age_2022,
    ('SoftClassificationSFCN', '/code/weights.h5'): sfcn_sm_brain_age_2022
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

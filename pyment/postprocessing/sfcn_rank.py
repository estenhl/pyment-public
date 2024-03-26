import numpy as np

def sfcn_rank_brain_age_2022(values: np.ndarray) -> np.ndarray:
    return 3 + np.sum(np.round(values), axis=-1)

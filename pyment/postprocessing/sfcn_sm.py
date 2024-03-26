import numpy as np


def sfcn_sm_brain_age_2022(values: np.ndarray) -> np.ndarray:
    return 3 + np.sum(np.arange(len(values)) * values)

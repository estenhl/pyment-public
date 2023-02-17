import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model

from pyment.models.utils import restrict_range


def test_restrict_range():
    data = np.arange(5).reshape((1, 5)).astype(np.float32)
    restricted = restrict_range(data, 1, 3)

    assert 1 == np.amin(restricted), \
        'restrict_range does not enforce lower bound'
    assert 3 == np.amax(restricted), \
        'restrict_range does not encforce upper bound'

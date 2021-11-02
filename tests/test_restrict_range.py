import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model

from pyment.models.utils import restrict_range


def test_restrict_range():
    restricted = restrict_range(np.arange(5), 1, 3)

    assert 1 == np.amin(restricted), ('restrict_range does not '
                                      'lower bound')
    assert 3 == np.amax(restricted), ('restrict_range does not '
                                      'upper bound')
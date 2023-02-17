"""Contains tests for testing the NiftiAugmenter."""

import os
import numpy as np

from pyment.data.augmenters import NiftiAugmenter


def test_mni_y_flip():
    """Tests that the augmenter applies a saggital flip (in MNI152
    space).
    """
    X = np.arange(3**3).reshape((3, 3, 3))
    flipped = NiftiAugmenter.fast_flip(X, [1, 0, 0])

    assert np.array_equal(X[::-1,:,:], flipped), \
        'NiftiAugmenter does not correctly flip in the saggital axis'

def test_mni_x_flip():
    """Tests that the augmenter applies a coronal flip (in MNI152
    space).
    """
    X = np.arange(3**3).reshape((3, 3, 3))
    flipped = NiftiAugmenter.fast_flip(X, [0, 1, 0])

    assert np.array_equal(X[:,::-1,:], flipped), \
        'NiftiAugmenter does not correctly flip in the coronal axis'

def test_mni_z_flip():
    """Tests that the augmenter applies a axial flip (in MNI152
    space).
    """
    X = np.arange(3**3).reshape((3, 3, 3))
    flipped = NiftiAugmenter.fast_flip(X, [0, 0, 1])

    assert np.array_equal(X[:,:,::-1], flipped), \
        'NiftiAugmenter does not correctly flip in the axial axis'

def test_augmenter_to_from_json():
    """Tests that an augmenter instantiated from json behaves
    equivalently as the augmenter that generated the json.
    """
    X = np.arange(3**3).reshape((3, 3, 3))
    source = NiftiAugmenter(flip_probabilities=[1, 1, 1])
    augmenter = NiftiAugmenter.from_json(source.json)

    assert np.array_equal(source(X), augmenter(X)), \
        ('NiftiAugmenter instantiated from json does not augment in the same '
         'fashion as the augmenter that generated the json')

def test_augmenter_to_from_jsonstring():
    """Tests that an augmenter instantiated from jsonstring behaves
    equivalently as the augmenter that generated the jsonstring.
    """
    X = np.arange(3**3).reshape((3, 3, 3))
    source = NiftiAugmenter(flip_probabilities=[1, 1, 1])
    augmenter = NiftiAugmenter.from_jsonstring(source.jsonstring)

    assert np.array_equal(source(X), augmenter(X)), \
        ('NiftiAugmenter instantiated from jsonstring does not augment in the '
         'same fashion as the augmenter that generated the jsonstring')

def test_augmenter_to_from_file():
    """Tests that an augmenter instantiated from file behaves
    equivalently as the augmenter that generated the file.
    """
    try:
        X = np.arange(3**3).reshape((3, 3, 3))
        source = NiftiAugmenter(flip_probabilities=[1, 1, 1])
        source.save('tmp.json')
        augmenter = NiftiAugmenter.from_file('tmp.json')

        assert np.array_equal(source(X), augmenter(X)), \
            ('NiftiAugmenter instantiated from file does not augment in the '
            'same fashion as the augmenter that generated the file')
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')

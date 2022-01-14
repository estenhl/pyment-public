"""Contains tests for the nifti preprocessor"""

import os
import numpy as np

from pyment.data.preprocessors import NiftiPreprocessor

def test_nifti_preprocessor_without_arguments():
    """Tests that a NiftiPreprocessor without any configuration doesn't
    make changes to the given data.
    """
    X = np.random.uniform((10, 10, 10, 10, 1))
    preprocessor = NiftiPreprocessor()
    preprocessed = preprocessor(X)

    assert np.array_equal(preprocessed, X), \
        'Using a NiftiPreprocessor without any arguments changes the data'

def test_nifti_preprocessor_sigma():
    """Tests that a NiftiPreprocessor applies the given sigma."""
    X = np.random.uniform((10, 10, 10, 10, 1))
    preprocessor = NiftiPreprocessor(sigma=5.)
    preprocessed = preprocessor(X)

    assert np.array_equal(preprocessed, X / 5), \
        'NiftiPreprocessor does not divide data by sigma'

def test_nifti_preprocessor_to_from_json():
    """Tests that a NiftiPreprocessor instantiated from json behaves
    equivalently to the one that generated the json."""
    X = np.random.uniform((10, 10, 10, 10, 1))
    source = NiftiPreprocessor(sigma=5.)
    preprocessor = NiftiPreprocessor.from_json(source.json)

    assert np.array_equal(source(X), preprocessor(X)), \
        ('NiftiPreprocessor instantiated from json does not behave '
         'equivalently as the NiftiPreprocessor that generated the json')

def test_nifti_preprocessor_to_from_jsonstring():
    """Tests that a NiftiPreprocessor instantiated from a jsonstring behaves
    equivalently to the one that generated the jsonstring."""
    X = np.random.uniform((10, 10, 10, 10, 1))
    source = NiftiPreprocessor(sigma=5.)
    preprocessor = NiftiPreprocessor.from_jsonstring(source.jsonstring)

    assert np.array_equal(source(X), preprocessor(X)), \
        ('NiftiPreprocessor instantiated from jsonstring does not behave '
         'equivalently as the NiftiPreprocessor that generated the jsonstring')

def test_nifti_preprocessor_to_from_file():
    """Tests that a NiftiPreprocessor instantiated from file behaves
    equivalently to the one that generated the file."""
    try:
        X = np.random.uniform((10, 10, 10, 10, 1))
        source = NiftiPreprocessor(sigma=5.)
        source.save('tmp.json')
        preprocessor = NiftiPreprocessor.from_file('tmp.json')

        assert np.array_equal(source(X), preprocessor(X)), \
            ('NiftiPreprocessor instantiated from file does not behave '
            'equivalently as the NiftiPreprocessor that generated the file')
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')

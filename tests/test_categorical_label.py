"""Tests for CategoricalLabel"""

import numpy as np

from pyment.labels import CategoricalLabel
from pyment.labels.missing_strategy import MissingStrategy

from utils import assert_exception


def test_categorical_label_initialize_encoding_str():
    """Tests whether a CategoricalLabel can be initialized with a
    string encoding.
    """
    label = CategoricalLabel('test', encoding='onehot')

    assert label.encoding == CategoricalLabel.Encoding.ONEHOT, \
        ('Setting CategoricalLabel encoding with a string in initializer does '
         'not set the property')

def test_categorical_label_initialize_encoding_enum():
    """Tests whether a CategoricalLabel can be initialized with a
    direct enum encoding.
    """
    label = CategoricalLabel('test', encoding=CategoricalLabel.Encoding.ONEHOT)

    assert label.encoding == CategoricalLabel.Encoding.ONEHOT, \
        ('Setting CategoricalLabel encoding with a string in initializer does '
         'not set the property')

def test_categorical_label_initialize_encoding_invalid():
    """Tests that initializating a CategoricalLabel with an invalid
    encoding raises an error.
    """
    assert_exception(lambda: CategoricalLabel('test', encoding='invalid'),
                     message=('Instantiating CategoricalLabel with an invalid '
                              'encoding does not raise an error'))

def test_categorical_label_fit_updates_mapping():
    """Tests that calling fit on a CategoricalLabel updates the mapping
    of the object.
    """
    label = CategoricalLabel('test', encoding='index')
    values = np.asarray(['B', 'C', 'A', 'A', 'C'])
    label.fit(values)

    assert {'A': 0, 'B': 1, 'C': 2} == label.mapping, \
        ('Calling fit on a CategoricalLabel does not update the '
         'mapping accordingly')

def test_categorical_label_fit_handles_int_nans():
    """Tests that calling fit on a CategoricalLabel with nans in the
    values is handled properly.
    """
    label = CategoricalLabel('test', encoding='index')
    values = np.asarray([3, 4, 2, 2, 3, np.nan])
    label.fit(values)

    assert {2: 0, 3: 1, 4: 2} == label.mapping, \
        'Calling fit on a CategoricalLabel does not properly remove nans'

def test_categorical_label_fit_handles_str_nans():
    """Tests that calling fit on a CategoricalLabel with nans in the
    values is handled properly.
    """
    label = CategoricalLabel('test', encoding='index')
    values = np.asarray(['B', 'C', 'A', 'A', 'C', np.nan])
    label.fit(values)

    assert {'A': 0, 'B': 1, 'C': 2} == label.mapping, \
        'Calling fit on a CategoricalLabel does not properly remove nans'

def test_categorical_label_fit_transform_index():
    """Tests that a CategoricalLabel is able to transform a set of
    values using an index encoding.
    """
    label = CategoricalLabel('test', encoding='index')
    values = np.asarray(['B', 'C', 'C', 'A', 'B'])
    transformed = label.fit_transform(values)

    assert np.array_equal([1, 2, 2, 0, 1], transformed), \
        'CategoricalLabel does not properly index encode variables'

def test_categorical_label_frequencies():
    """Tests that CategoricalLabel.frequencies is update by fit."""
    label = CategoricalLabel('test', encoding='index')
    values = np.asarray(['B', 'C', 'C', 'A', 'B'])
    label.fit(values)

    assert {'A': 1, 'B': 2, 'C': 2} == label.frequencies, \
        'CategoricalLabel.frequencies is not set properly by fit'

def test_categorical_label_reference():
    """Tests that CategoricalLabel.reference returns the correct
    value.
    """
    label = CategoricalLabel('test', encoding='index')
    values = np.asarray(['B', 'C', 'C', 'A', 'B', 'B'])
    label.fit(values)

    assert 'C' == label.reference, \
        ('CategoricalLabel.reference does not return the correct '
         'reference level')

def test_categorical_label_fill_with_reference():
    """Tests that a CategoricalLabel is able to fill missing values
    with the reference level.
    """
    label = CategoricalLabel('test', encoding='index',
                             missing_strategy=MissingStrategy.REFERENCE_FILL)
    values = np.asarray(['B', 'C', 'C', 'A', 'B', 'C', np.nan])
    transformed = label.fit_transform(values)

    assert np.array_equal([1, 2, 2, 0, 1, 2, 2], transformed), \
        ('CategoricalLabel does not properly fill nans with the '
         'reference level')

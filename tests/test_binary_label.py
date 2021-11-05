import os
import numpy as np

from pyment.labels import load_label_from_jsonfile, BinaryLabel, MissingStrategy

from utils import assert_exception


def test_binary_label_is_not_fitted_initially():
    label = BinaryLabel('test')

    assert not label.is_fitted, 'BinaryLabel is fitted from initialization'

def test_binary_label_is_fitted():
    label = BinaryLabel('test')
    values = np.asarray(['B', 'A', 'B', 'A', 'B'])
    label.fit(values)

    assert label.is_fitted, \
        'BinaryLabel does not report as being fitted after calling fit'

def test_binary_label_fit_encoding():
    label = BinaryLabel('test')
    values = np.asarray(['B', 'A', 'B', 'A', 'B'])
    label.fit(values)

    assert label.encoding is not None, \
        'BinaryLabel does not have an encoding after calling fit'
    assert {'A': 0, 'B': 1} == label.encoding, \
        'BinaryLabel does not encode values correctly'

def test_binary_label_given_encoding():
    label = BinaryLabel('test', encoding={'B': 0, 'A': 1})
    values = np.asarray(['B', 'A', 'B', 'A', 'B'])
    label.fit(values)

    assert {'B': 0, 'A': 1} == label.encoding, \
        'BinaryLabel does not use explicitly given encoding'

def test_binary_label_allowed_values():
    label = BinaryLabel('test', allowed=set(['A', 'B']))
    values = np.asarray(['B', 'A', 'C', 'A', 'B'])
    label.fit(values)

    assert {'A': 0, 'B': 1} == label.encoding, \
        ('BinaryLabel does not use correct encoding given a set of allowed '
         'values')

def test_binary_label_transform_before_fit():
    label = BinaryLabel('test', encoding={'A': 0, 'B': 1})
    values = np.asarray(['B', 'A', 'B', 'A', 'B'])

    assert_exception(label.transform, args=[values], exception=ValueError,
                     message=('Calling transform on an unfitted BinaryLabel '
                              'does not raise an exception'))

def test_binary_label_encoding():
    label = BinaryLabel('test', encoding={'A': 0, 'B': 1})
    values = np.asarray(['B', 'A', 'B', 'A', 'B'])
    label.fit(values)
    values = label.transform(values)

    assert np.array_equal([1, 0, 1, 0, 1], values), \
        'BinaryLabel does not encode values correctly'

def test_binary_label_fit_transform():
    label = BinaryLabel('test', encoding={'A': 0, 'B': 1})
    values = np.asarray(['B', 'A', 'B', 'A', 'B'])
    values = label.fit_transform(values)

    assert np.array_equal([1, 0, 1, 0, 1], values), \
        'BinaryLabel does not encode values correctly'

def test_binary_label_fit_excessive_levels():
    label = BinaryLabel('test')
    values = np.asarray(['A', 'B', 'C'])
    
    assert_exception(label.fit, args=[values], exception=AssertionError,
                     message=('Calling fit on a BinaryLabel with >2 levels '
                              'and not explicitly setting allowed values or '
                              'encoding does not raise an error'))

def test_binary_label_fit_excessive_allowed_levels():
    assert_exception(BinaryLabel, args=['test'], 
                     kwargs={'allowed': set(['A', 'B', 'C'])}, 
                     exception=AssertionError,
                     message=('Instantiating a BinaryLabel with >2 allowed '
                              'levels does not raise an error'))

def test_binary_label_fit_excessive_encoding_levels():
    assert_exception(BinaryLabel, args=['test'], 
                     kwargs={'encoding': {'A': 0, 'B': 1, 'C': 2}}, 
                     exception=AssertionError,
                     message=('Instantiating a BinaryLabel with >2 encoding '
                              'levels does not raise an error'))

def test_binary_label_unknown_values_encoding():
    label = BinaryLabel('test', encoding={'A': 0, 'B': 1})
    values = np.asarray(['A', 'B', 'C'])
    values = label.fit_transform(values)

    expected = np.asarray([0, 1, np.nan])

    assert np.array_equal(expected, values, equal_nan=True), \
        'BinaryLabel does not encode unknown variables correctly'

def test_binary_label_unknown_values_allowed():
    label = BinaryLabel('test', allowed=set(['A', 'B']))
    values = np.asarray(['A', 'B', 'C'])
    values = label.fit_transform(values)

    expected = np.asarray([0., 1., np.nan])

    assert np.array_equal(expected, values, equal_nan=True), \
        'BinaryLabel does not encode unknown variables correctly'

def test_binary_label_frequencies():
    label = BinaryLabel('test')
    values = np.asarray(['B', 'A', 'B', 'A', 'B'])
    label.fit(values)

    assert {'A': 2/5, 'B': 3/5} == label.frequencies, \
        'BinaryLabel does not report the correct frequencies'

def test_binary_label_frequencies_with_unknowns():
    label = BinaryLabel('test', allowed=set(['A', 'B']))
    values = np.asarray(['C', 'B', 'A', 'C', 'B', 'A', 'B', 'C'])
    label.fit(values)

    assert {'A': 2/5, 'B': 3/5} == label.frequencies, \
        'BinaryLabel does not report the correct frequencies'

def test_binary_label_missing_strategy_allow():
    label = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.ALLOW)
    values = np.asarray(['C', 'B', 'A', 'C', 'B', 'A', 'B', 'C'])
    values = label.fit_transform(values)

    expected = np.asarray([np.nan, 1., 0., np.nan, 1., 0., 1., np.nan])

    assert np.array_equal(expected, values, equal_nan=True), \
        'BinaryLabel with allow strategy does not allow NAs'

def test_binary_label_missing_strategy_mean_fill():
    label = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    values = np.asarray(['C', 'B', 'A', 'C', 'B', 'A', 'B', 'C'])
    values = label.fit_transform(values)

    expected = np.asarray([3/5, 1., 0., 3/5, 1., 0., 1., 3/5])

    assert np.array_equal(expected, values, equal_nan=True), \
        'BinaryLabel with mean_fill strategy does not correctly encode NAs'

def test_binary_label_missing_strategy_mean_fill():
    label = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.CENTRE_FILL)
    values = np.asarray(['C', 'B', 'A', 'C', 'B', 'A', 'B', 'C'])
    values = label.fit_transform(values)

    expected = np.asarray([0.5, 1., 0., 0.5, 1., 0., 1., 0.5])

    assert np.array_equal(expected, values, equal_nan=True), \
        'BinaryLabel with centre_fill strategy does not correctly encode NAs'

def test_binary_label_json():
    label = BinaryLabel('test', missing_strategy=MissingStrategy.MEAN_FILL)

    assert 'name' in label.json, \
        'BinaryLabel.json does not contain a field for name'
    assert 'test' == label.json['name'], \
        'BinaryLabel.json contains the wrong name'
    assert 'missing_strategy' in label.json, \
        'BinaryLabel.json does not contain a field for missing strategy'
    assert 'mean_fill' == label.json['missing_strategy'], \
        'BinaryLabel.json contains the wrong missing strategy'

def test_binary_label_json_fit():
    label = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label.fit(np.asarray(['A', 'B', 'C', 'B', 'B']))

    assert 'fit' in label.json, \
        'Fitted BinaryLabel.json does not contain a field for the fit'
    
    fit = label.json['fit']

    assert 'encoding' in fit, \
        ('Fitted BinaryLabel.json does not contain a field for the fitted '
         'encoding')
    assert {'A': 0, 'B': 1} == fit['encoding'], \
        'Fitted BinaryLabel.json contains the wrong encoding'
    assert 'frequencies' in fit, \
        ('Fitted BinaryLabel.json does not contain a field for the fitted '
         'frequencies')
    assert {'A': 1/4, 'B': 3/4} == fit['frequencies'], \
        'Fitted BinaryLabel.json contains the wrong frequencies'

def test_binary_label_equality():
    label1 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label2 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)

    assert label1 == label2, 'Equal BinaryLabels are not considered equal'

def test_binary_label_unequal_name():
    label1 = BinaryLabel('test1', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label2 = BinaryLabel('test2', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)

    assert label1 != label2, \
        'BinaryLabels with different names are considered equal'

def test_binary_label_unequal_name():
    label1 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label2 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.ALLOW)

    assert label1 != label2, \
        'BinaryLabels with different missing strategies are considered equal'

def test_binary_label_unequal_fit():
    label1 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label1.fit(np.asarray(['A', 'B']))
    label2 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)

    assert label1 != label2, \
        'BinaryLabels where one is fit and one is not fit is considered equal'

def test_binary_label_unequal_fit_values():
    label1 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label1.fit(np.asarray(['A', 'B', 'B']))
    label2 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label2.fit(np.asarray(['A', 'A', 'B']))

    assert label1 != label2, \
        'BinaryLabels fit with different values are considered equal'

def test_binary_label_unequal_allowed():
    label1 = BinaryLabel('test', allowed=set(['A', 'B']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label2 = BinaryLabel('test', allowed=set(['A', 'C']), 
                        missing_strategy=MissingStrategy.MEAN_FILL)

    assert label1 != label2, \
        'BinaryLabels with different allowed values are considered equal'

def test_binary_label_unequal_allowed():
    label1 = BinaryLabel('test', encoding={'A': 0, 'B': 1}, 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label2 = BinaryLabel('test', encoding={'A': 1, 'B': 0}, 
                        missing_strategy=MissingStrategy.MEAN_FILL)

    assert label1 != label2, \
        'BinaryLabels with different encodings are considered equal'

def test_binary_label_to_from_json():
    label1 = BinaryLabel('test', encoding={'A': 0, 'B': 1}, 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    label2 = BinaryLabel.from_json(label1.json)

    assert label1 == label2, \
        'BinaryLabel to and from json does not produce an equivalent object'

def test_fitted_binary_label_to_from_json():
    label1 = BinaryLabel('test', encoding={'A': 0, 'B': 1}, 
                        missing_strategy=MissingStrategy.MEAN_FILL)
    values1 = label1.fit_transform(np.asarray(['A', 'B', 'B', 'C', 'B']))
    label2 = BinaryLabel.from_json(label1.json)

    assert label1 == label2, \
        'BinaryLabel to and from json does not produce an equivalent object'

    values2 = label2.transform(np.asarray(['A', 'B', 'B', 'C', 'B']))

#     assert np.array_equal(values1, values2), \
#         'BinaryLabel to and from json does not produce the same encoded values'

# def test_fitted_binary_label_to_from_jsonstring():
#     label1 = BinaryLabel('test', encoding={'A': 0, 'B': 1}, 
#                         missing_strategy=MissingStrategy.MEAN_FILL)
#     values1 = label1.fit_transform(np.asarray(['A', 'B', 'B', 'C', 'B']))
#     label2 = BinaryLabel.from_jsonstring(label1.jsonstring)

#     assert label1 == label2, \
#         'BinaryLabel to and from json does not produce an equivalent object'

#     values2 = label2.transform(np.asarray(['A', 'B', 'B', 'C', 'B']))

#     assert np.array_equal(values1, values2), \
#         'BinaryLabel to and from json does not produce the same encoded values'

# def test_binary_label_save_load():
#     try:
#         label1 = BinaryLabel('test', encoding={'A': 0, 'B': 1}, 
#                             missing_strategy=MissingStrategy.MEAN_FILL)
#         label1.save('tmp.json')
#         label2 = load_label_from_jsonfile('tmp.json')

#         assert label1 == label2, \
#             'BinaryLabel save and load does not produce an equivalent object'
#     finally:
#         if os.path.isfile('tmp.json'):
#             os.remove('tmp.json')

# def test_equal_string():
#     assert BinaryLabel('test') != 'test', \
#         'BinaryLabel is considered equal as string with the same value'
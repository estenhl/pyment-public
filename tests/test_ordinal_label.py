import os
import numpy as np

from pyment.labels import load_label_from_jsonfile, Label, OrdinalLabel


def test_ordinal_label_name():
    label = OrdinalLabel('test')

    assert 'test' == label.name, 'OrdinalLabel does not get the correct name'

def test_ordinal_label_from_type():
    label = Label.from_type('ordinal', name='test')

    assert isinstance(label, OrdinalLabel), \
        'Label.from_type(ordinal) does not return an OrdinalLabel'

def test_ordinal_label_ranges():
    label = OrdinalLabel('test', ranges={'low': (0, 10), 'medium': (11, 53),
                                         'high': (54, 100)})

    values = label.fit_transform(np.asarray(['low', 'medium', 'high']))

    assert np.array_equal([5, 32, 77], values), \
        'OrdinalLabel does not transform ranged variables to their mean'

def test_ordinal_label_unknown():
    label = OrdinalLabel('test', ranges={'low': (0, 10), 'medium': (11, 53),
                                         'high': (54, 100)})

    values = label.fit_transform(np.asarray(['low', 'medium', 'high', 'x']))

    assert np.array_equal([5, 32, 77, np.nan], values, equal_nan=True), \
        'OrdinalLabel does not handle unknown values'

def test_ordinal_label_applies_mu_sigma():
    label = OrdinalLabel('test', ranges={'low': (0, 10), 'medium': (11, 53),
                                         'high': (54, 100)}, 
                         mu=10, sigma=2)

    values = label.fit_transform(np.asarray(['low', 'medium', 'high', 'x']))

    assert np.array_equal([-2.5, 11, 33.5, np.nan], values, equal_nan=True), \
        'OrdinalLabel does not apply mu and sigma'

def test_ordinal_label_standardize():
    label = OrdinalLabel('test', ranges={'low': (0, 10), 'medium': (11, 53),
                                         'high': (54, 100)}, 
                         standardize=True)

    values = label.fit_transform(np.asarray(['low', 'medium', 'high', 'x']))

    assert abs(np.nanmean(values)) < 1e-5, \
        'OrdinalLabel with standardize does not have mean 0'
    assert abs(1 - np.nanstd(values)) < 1e-5, \
        'OrdinalLabel with standardize does not have stddev 1'

def test_ordinal_label_save_load():
    try:
        label = OrdinalLabel('test', ranges={'low': (0, 10), 'medium': (11, 53),
                                            'high': (54, 100)}, 
                            standardize=True)

        label.fit(np.asarray(['low', 'medium', 'high', 'x']))
        label.save('tmp.json')
        label = load_label_from_jsonfile('tmp.json')

        values = label.transform(np.asarray(['low', 'medium', 'high', 'x', 
                                             'low', 'high']))
        
        expected = np.asarray([-1.11116, -0.20203, 1.31319, np.nan,
                               -1.11116, 1.31319])

        assert np.allclose(expected, values, 1e-5, equal_nan=True), \
            'OrdinalLabel save and load does not produce correct values'
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')
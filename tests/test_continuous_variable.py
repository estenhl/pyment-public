import os
import numpy as np

from pyment.labels import load_label_from_jsonfile, ContinuousLabel, \
                          MissingStrategy

from utils import assert_exception


def test_continuous_label_name():
    label = ContinuousLabel('test')

    assert 'test' == label.name, \
        'ContinuousLabel does not get the correct name'

def test_continuous_label_mu():
    label = ContinuousLabel('test', mu=1)
    values = label.fit_transform(np.asarray([1, 2, 3]))

    assert np.array_equal([0, 1, 2], values), \
        'ContinuousLabel does not subtract mu'

def test_continuous_label_sigma():
    label = ContinuousLabel('test', sigma=2)
    values = label.fit_transform(np.asarray([2, 4, 6]))

    assert np.array_equal([1, 2, 3], values), \
        'ContinuousLabel does not divide by sigma'

def test_continuous_label_normalize():
    label = ContinuousLabel('test', normalize=True)
    values = label.fit_transform(np.asarray([2, 4, 6]))

    assert np.array_equal([0, 0.5, 1], values), \
        'ContinuousLabel does not normalize properly'

def test_continuous_label_standardize():
    label = ContinuousLabel('test', standardize=True)
    values = label.fit_transform(np.asarray([2, 4, 6]))

    assert np.abs(np.mean(values)) < 1e-5, \
        'ContinuousLabel does not standardize properly'
    assert np.abs(1 - np.std(values)) < 1e-5, \
        'ContinuousLabel does not standardize properly'

def test_continuous_label_mu_nan():
    label = ContinuousLabel('test', mu=1)
    values = label.fit_transform(np.asarray([1, 2, 3, np.nan]))

    assert np.array_equal([0, 1, 2, np.nan], values, equal_nan=True), \
        'ContinuousLabel does not subtract mu'

def test_continuous_label_sigma_nan():
    label = ContinuousLabel('test', sigma=2)
    values = label.fit_transform(np.asarray([2, 4, 6, np.nan]))

    assert np.array_equal([1, 2, 3, np.nan], values, equal_nan=True), \
        'ContinuousLabel does not divide by sigma'

def test_continuous_label_normalize_nan():
    label = ContinuousLabel('test', normalize=True)
    values = label.fit_transform(np.asarray([2, 4, 6, np.nan]))

    assert np.array_equal([0, 0.5, 1, np.nan], values, equal_nan=True), \
        'ContinuousLabel does not normalize properly'

def test_continuous_label_standardize_nan():
    label = ContinuousLabel('test', standardize=True)
    values = label.fit_transform(np.asarray([2, 4, 6, np.nan]))

    assert np.abs(np.nanmean(values)) < 1e-5, \
        'ContinuousLabel does not standardize properly'
    assert np.abs(1 - np.nanstd(values)) < 1e-5, \
        'ContinuousLabel does not standardize properly'

def test_continuous_label_mean():
    label = ContinuousLabel('test', mu=1, sigma=3)
    label.fit(np.asarray([4, 10, 16, np.nan]))
    
    assert 3 == label.mean, \
        'ContinuousLabel.mean is not set properly'

def test_continuous_label_floor():
    label = ContinuousLabel('test', floor=1)
    values = label.fit(np.asarray([0, 1, 2, np.nan]), transform=True)

    assert np.array_equal([1, 1, 2, np.nan], values, equal_nan=True), \
        'ContinuousLabel does not apply floor'

def test_continuous_label_ceil():
    label = ContinuousLabel('test', ceil=1)
    values = label.fit(np.asarray([0, 1, 2, np.nan]), transform=True)

    assert np.array_equal([0, 1, 1, np.nan], values, equal_nan=True), \
        'ContinuousLabel does not apply ceil'

def test_continuous_label_stddev():
    label = ContinuousLabel('test')
    values = np.random.normal(loc=0, scale=2, size=100)
    label.fit(values)

    assert label.stddev == np.std(values), \
        'ContinuousLabel.stddev is not set properly'

def test_continuous_label_mean_fill():
    label = ContinuousLabel('test', normalize=True, 
                            missing_strategy=MissingStrategy.MEAN_FILL)
    values = label.fit_transform(np.asarray([1, 2, 3, np.nan]))

    assert np.array_equal([0, 0.5, 1, 0.5], values), \
        'ContinuousLabel does not mean fill correctly'

def test_continuous_label_fit_nans():
    label = ContinuousLabel('test')

    values = np.asarray([np.nan, np.nan, np.nan])
    assert_exception(label.fit, args=[values], exception=ValueError,
                     message=('ContinuousLabel does not raise an exception if '
                              'asked to fit on all nans'))

def test_continuous_label_zero_fill():
    label = ContinuousLabel('test', normalize=True, 
                            missing_strategy=MissingStrategy.ZERO_FILL)
    values = label.fit_transform(np.asarray([1, 2, 3, np.nan]))

    assert np.array_equal([0, 0.5, 1, 0], values), \
        'ContinuousLabel does not zero fill correctly'

def test_continuous_label_sample_nans():
    np.random.seed(42)

    label = ContinuousLabel('test', missing_strategy=MissingStrategy.SAMPLE)
    values = np.random.normal(loc=0, scale=1, size=500)
    values[250:] = np.nan
    values = label.fit_transform(values)


    assert not any(np.isnan(values)), \
        'ContinuousLabel with missing strategy sample does not replace nans'
    assert abs(np.mean(values)) < 1e-1, \
        ('ContinuousLabel with missing strategy sample does not utilize '
         'correct mean')
    assert abs(1 - np.std(values)) < 1e-1, \
        ('ContinuousLabel with missing strategy sample does not utilize '
         'correct stddev')

def test_continuous_label_save_load():
    try:
        label1 = ContinuousLabel('test', standardize=True, 
                                missing_strategy=MissingStrategy.MEAN_FILL)
        values = np.random.uniform(0, 1, 100)
        values[np.random.permutation(np.arange(100))[:50]] = np.nan
        transformed1 = label1.fit_transform(values)

        label1.save('tmp.json')
        label2 = load_label_from_jsonfile('tmp.json')
        transformed2 = label2.transform(values)

        assert label1 == label2, \
            ('ContinuousLabel save and load does not instantiate an '
             'equivalent object')
        assert np.allclose(transformed1, transformed2, 1e-5), \
            ('ContinuousLabel save and load does not produce equal '
             'transformed values')
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')

def test_continuous_label_reinitialization():
        label = ContinuousLabel('test', standardize=True, 
                                 missing_strategy=MissingStrategy.MEAN_FILL)
        values = np.random.uniform(0, 1, 100)
        values[np.random.permutation(np.arange(100))[:50]] = np.nan
        label.fit_transform(values)

        assert_exception(ContinuousLabel, args=['test'], 
                         kwargs={'mu': 2, 'fit': label._fit}, 
                         exception=ValueError,
                         message=('Initializing a ContinuousLabel with a '
                                  'previous fit and new configuration does '
                                  'not raise an error'))

def test_continuous_label_min():
    label = ContinuousLabel('test', mu=1, sigma=3)
    label.fit(np.asarray([4, 10, 16, np.nan]))
    
    assert 1 == label.min, \
        'ContinuousLabel.min is not set properly'

def test_continuous_label_max():
    label = ContinuousLabel('test', mu=1, sigma=3)
    label.fit(np.asarray([4, 10, 16, np.nan]))
    
    assert 5 == label.max, \
        'ContinuousLabel.max is not set properly'
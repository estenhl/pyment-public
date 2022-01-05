from pyment.labels import CategoricalLabel

from utils import assert_exception


def test_categorical_label_initialize_encoding_str():
    label = CategoricalLabel('test', encoding='onehot')

    assert label.encoding == CategoricalLabel.Encoding.ONEHOT, \
        ('Setting CategoricalLabel encoding with a string in initializer does '
         'not set the property')

def test_categorical_label_initialize_encoding_str():
    label = CategoricalLabel('test', encoding=CategoricalLabel.Encoding.ONEHOT)

    assert label.encoding == CategoricalLabel.Encoding.ONEHOT, \
        ('Setting CategoricalLabel encoding with a string in initializer does '
         'not set the property')

def test_categorical_label_initialize_encoding_invalid():
    assert_exception(lambda: CategoricalLabel('test', encoding='invalid'),
                     message=('Instantiating CategoricalLabel with an invalid '
                              'encoding does not raise an error'))

def test_categorical_label_is_fitted():
    label = CategoricalLabel('test', encoding='onehot')

    assert not label.is_fitted, 'CategoricalLabel is fitted by default'

    label = CategoricalLabel('test', encoding='onehot',
                             mapping={'A': 0, 'B': 1, 'C': 2})

    assert label.is_fitted, \
        'CategoricalLabel with given mapping is not fitted by default'

def test_categorical_label_is_fitted():
    label = CategoricalLabel('test', encoding='onehot')

    assert not label.is_fitted, 'CategoricalLabel is fitted by default'

    label = CategoricalLabel('test', encoding='onehot',
                             mapping={'A': 0, 'B': 1, 'C': 2})

    assert label.is_fitted, \
        'CategoricalLabel with given mapping is not fitted by default'

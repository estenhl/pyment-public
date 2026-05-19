import pandas as pd
import pytest

from pyment.data.dataset import Dataset


class _ConcreteDataset(Dataset):
    def to_tensorflow_generator(self, *args, **kwargs):
        pass


@pytest.fixture
def dataset():
    labels = pd.DataFrame(
        {
            'image_path': ['/a', '/b', '/c', '/d'],
            'target': [0, 0, 1, 1],
        }
    )
    return _ConcreteDataset(labels=labels, target='target')


def test_class_weights_none(dataset):
    dataset.class_weights = None

    assert dataset.class_weights is None, (
        'Expected class_weights to return None when set to None'
    )


def test_class_weights_balanced(dataset):
    dataset.class_weights = 'balanced'

    assert dataset.class_weights == {0: 1.0, 1: 1.0}, (
        'Expected class_weights to return equal weights for a balanced '
        'class distribution'
    )


def test_class_weights_explicit_dict(dataset):
    dataset.class_weights = {0: 2.0, 1: 0.5}

    assert dataset.class_weights == {0: 2.0, 1: 0.5}, (
        'Expected class_weights to return the given dict weights'
    )


def test_class_weights_invalid_raises(dataset):
    with pytest.raises(ValueError, match='Invalid class_weights value'):
        dataset.class_weights = 'invalid'

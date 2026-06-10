import pandas as pd
import pytest
import tensorflow as tf

from pyment.configurations.data_split_configuration import (
    DataSplitConfiguration,
)
from pyment.data.dataset import Dataset


class SimpleDataset(Dataset):
    def to_tensorflow_generator(self, *args, **kwargs) -> tf.data.Dataset:
        paths = self.labels['image_path'].values
        raw_targets = self.labels[self.target].values
        if self.target_encoder:
            targets = [self.target_encoder(t) for t in raw_targets]
        else:
            targets = raw_targets
        return tf.data.Dataset.from_tensor_slices((paths, targets))


def make_dataset(n, target_encoder=None):
    labels = pd.DataFrame(
        {
            'image_path': [f'/images/{i}.nii.gz' for i in range(n)],
            'age': [float(i) for i in range(n)],
        }
    )
    return SimpleDataset(
        labels=labels, target='age', target_encoder=target_encoder
    )


def encode(x):
    return int(x > 5)


@pytest.mark.parametrize(
    'n, fraction',
    [(10, 0.8), (10, 0.5), (7, 0.7)],
    ids=['80pct', '50pct', '70pct-uneven'],
)
def test_split_sizes(n, fraction):
    dataset = make_dataset(n)
    config = DataSplitConfiguration(training_fraction=fraction)
    train, val = config.split(dataset)

    expected_train = int(fraction * n)
    assert len(train) == expected_train
    assert len(val) == n - expected_train


def test_split_covers_all_samples():
    dataset = make_dataset(10)
    train, val = DataSplitConfiguration(training_fraction=0.8).split(dataset)

    assert len(train) + len(val) == len(dataset)


def test_split_preserves_image_label_pairing():
    dataset = make_dataset(10)
    train, val = DataSplitConfiguration(training_fraction=0.6).split(dataset)

    for path, label in train.to_tensorflow_generator():
        idx = int(path.numpy().decode().split('/')[-1].replace('.nii.gz', ''))
        assert label.numpy() == pytest.approx(float(idx))

    for path, label in val.to_tensorflow_generator():
        idx = int(path.numpy().decode().split('/')[-1].replace('.nii.gz', ''))
        assert label.numpy() == pytest.approx(float(idx))


def test_split_retains_target_encoder():
    dataset = make_dataset(10, target_encoder=encode)
    train, val = DataSplitConfiguration(training_fraction=0.8).split(dataset)

    assert train.target_encoder is encode
    assert val.target_encoder is encode


def test_split_target_encoder_applied_in_generator():
    dataset = make_dataset(10, target_encoder=encode)
    train, val = DataSplitConfiguration(training_fraction=0.5).split(dataset)

    for path, label in train.to_tensorflow_generator():
        idx = int(path.numpy().decode().split('/')[-1].replace('.nii.gz', ''))
        assert label.numpy() == encode(float(idx))

    for path, label in val.to_tensorflow_generator():
        idx = int(path.numpy().decode().split('/')[-1].replace('.nii.gz', ''))
        assert label.numpy() == encode(float(idx))

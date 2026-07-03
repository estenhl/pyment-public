import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from pyment.data import NiftiDataset


def make_mgh(path: str, shape: tuple, value: float = 1.0) -> None:
    data = np.full(shape, value, dtype=np.float32)
    nib.save(nib.MGHImage(data, affine=np.eye(4)), path)


@pytest.fixture
def images_folder(tmp_path):
    make_mgh(str(tmp_path / 'sub-01.mgz'), (4, 5, 6), value=1.0)
    make_mgh(str(tmp_path / 'sub-02.mgz'), (5, 5, 5), value=2.0)
    return tmp_path


@pytest.fixture
def labels_path(tmp_path, images_folder):
    labels = pd.DataFrame(
        {
            'image_id': ['sub-01', 'sub-02', 'sub-03'],
            'target': [0, 1, 1],
        }
    )
    path = tmp_path / 'labels.csv'
    labels.to_csv(path, index=False)
    return str(path)


def test_from_flat_folder_matches_files_to_image_id(images_folder, labels_path):
    dataset = NiftiDataset.from_flat_folder(
        images_path=str(images_folder), labels_path=labels_path, target='target'
    )

    paths = sorted(dataset.labels['image_path'].tolist())
    assert paths == sorted(
        [
            str(images_folder / 'sub-01.mgz'),
            str(images_folder / 'sub-02.mgz'),
        ]
    ), (
        'Expected from_flat_folder to map each image_id to the matching '
        'file path in images_path'
    )


def test_from_flat_folder_drops_rows_with_no_matching_file(
    images_folder, labels_path
):
    dataset = NiftiDataset.from_flat_folder(
        images_path=str(images_folder), labels_path=labels_path, target='target'
    )

    assert 'sub-03' not in dataset.labels['image_id'].tolist(), (
        'Expected from_flat_folder to drop rows with no matching file in '
        'images_path'
    )


def test_from_flat_folder_drops_rows_with_missing_target(
    images_folder, tmp_path
):
    labels = pd.DataFrame(
        {
            'image_id': ['sub-01', 'sub-02'],
            'target': [0, None],
        }
    )
    path = tmp_path / 'labels.csv'
    labels.to_csv(path, index=False)

    dataset = NiftiDataset.from_flat_folder(
        images_path=str(images_folder), labels_path=str(path), target='target'
    )

    assert dataset.labels['image_id'].tolist() == ['sub-01'], (
        'Expected from_flat_folder to drop rows with a missing target value'
    )


def test_to_tensorflow_generator_pads_to_target_shape(
    images_folder, labels_path
):
    dataset = NiftiDataset.from_flat_folder(
        images_path=str(images_folder), labels_path=labels_path, target='target'
    )

    generator = dataset.to_tensorflow_generator(
        batch_size=2, target_shape=(8, 8, 8)
    )
    images, targets = next(iter(generator))

    assert tuple(images.shape) == (2, 8, 8, 8), (
        'Expected to_tensorflow_generator to pad images of differing '
        'shapes up to a common target_shape before batching'
    )
    assert sorted(targets.numpy().tolist()) == [0, 1], (
        'Expected to_tensorflow_generator to yield targets matching the '
        'labels for the batched images'
    )


def test_to_tensorflow_generator_preserves_image_content(
    images_folder, labels_path
):
    dataset = NiftiDataset.from_flat_folder(
        images_path=str(images_folder), labels_path=labels_path, target='target'
    )

    generator = dataset.to_tensorflow_generator(
        batch_size=2, target_shape=(8, 8, 8)
    )
    images, _ = next(iter(generator))

    values = {round(float(v), 1) for v in np.unique(images.numpy())}
    assert values == {0.0, 1.0, 2.0}, (
        'Expected to_tensorflow_generator to preserve original voxel '
        'values (1.0, 2.0) alongside zero-padding (0.0)'
    )

import nibabel as nib
import numpy as np
import pytest

from pyment.data.utils import ensure_fastsurfer_crop_exists


@pytest.fixture
def mri_folder(tmp_path):
    (tmp_path / 'mri').mkdir()
    return tmp_path


@pytest.fixture
def mri_folder_with_files(mri_folder):
    data = np.ones((10, 10, 10), dtype=np.float32)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    image = nib.Nifti1Image(data, affine=affine)

    nib.save(image, str(mri_folder / 'mri' / 'orig.mgz'))
    nib.save(image, str(mri_folder / 'mri' / 'mask.mgz'))

    return mri_folder


def test_returns_true_if_crop_already_exists(mri_folder):
    (mri_folder / 'mri' / 'crop.mgz').touch()

    assert ensure_fastsurfer_crop_exists(str(mri_folder)), (
        'Expected ensure_fastsurfer_crop_exists to return True when the '
        'crop already exists'
    )


def test_returns_false_if_orig_is_missing(mri_folder):
    assert not ensure_fastsurfer_crop_exists(str(mri_folder)), (
        'Expected ensure_fastsurfer_crop_exists to return False when '
        'orig.mgz is missing'
    )


def test_returns_false_if_mask_is_missing(mri_folder):
    (mri_folder / 'mri' / 'orig.mgz').touch()

    assert not ensure_fastsurfer_crop_exists(str(mri_folder)), (
        'Expected ensure_fastsurfer_crop_exists to return False when '
        'mask.mgz is missing'
    )


def test_creates_crop_and_returns_true(mri_folder_with_files):
    result = ensure_fastsurfer_crop_exists(str(mri_folder_with_files))
    crop_path = mri_folder_with_files / 'mri' / 'crop.mgz'

    assert result, (
        'Expected ensure_fastsurfer_crop_exists to return True when '
        'both source files are present'
    )
    assert crop_path.exists(), (
        'Expected ensure_fastsurfer_crop_exists to yield crop.mgz when '
        'both source files are present'
    )

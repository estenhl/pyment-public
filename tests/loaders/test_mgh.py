import os
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import tensorflow as tf
from nibabel.spatialimages import SpatialImage

from pyment.loaders.mgh import load_mgh

fixtures_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'fixtures'
)
fastsurfer_path = os.path.join(fixtures_path, 'fastsurfer')


def test_load_mgh_matches_nibabel():
    path = os.path.join(fastsurfer_path, 'esten.mgh')
    result = load_mgh(tf.constant(path)).numpy()

    nib_image = cast(SpatialImage, nib.load(path))
    expected = np.asarray(nib_image.dataobj).astype(np.float32)

    np.testing.assert_allclose(result, expected)


def test_load_mgz_matches_nibabel():
    path = os.path.join(fastsurfer_path, 'esten.mgz')
    result = load_mgh(tf.constant(path)).numpy()

    nib_image = cast(SpatialImage, nib.load(path))
    expected = np.asarray(nib_image.dataobj).astype(np.float32)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    'dtype,np_dtype',
    [
        (np.uint8, np.uint8),
        (np.int16, np.int16),
        (np.int32, np.int32),
    ],
)
@pytest.mark.parametrize('ext', ['.mgh', '.mgz'])
def test_load_dtype_matches_nibabel(tmp_path, dtype, np_dtype, ext):
    data = np.arange(800, dtype=np_dtype).reshape(10, 8, 10)
    img = nib.MGHImage(data, np.eye(4))
    path = str(tmp_path / f'test{ext}')
    nib.save(img, path)

    result = load_mgh(tf.constant(path)).numpy()
    nib_image = cast(SpatialImage, nib.load(path))
    expected = np.asarray(nib_image.dataobj).astype(np.float32)

    np.testing.assert_allclose(result, expected)

import numpy as np
import nibabel as nib

from pyment.preprocessing.crop import calculate_bounds, crop_nifti_image_if_necessary


def test_calculate_bounds_centers_on_brain_with_left_adjustment():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    image[1:3, 1:3, 1:3] = 1.0
    target_shape = (6, 6, 6)

    idx = calculate_bounds(image, target_shape)

    assert idx == [slice(0, 6), slice(0, 6), slice(0, 6)]


def test_calculate_bounds_centers_on_brain_with_right_adjustment():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    image[8:9, 8:9, 8:9] = 1.0
    target_shape = (6, 6, 6)

    idx = calculate_bounds(image, target_shape)

    assert idx == [slice(4, 10), slice(4, 10), slice(4, 10)]


def test_crop_nifti_image_if_necessary_keeps_brain_region():
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[7:9, 7:9, 7:9] = 2.0
    image = nib.Nifti1Image(data, affine=np.eye(4))
    target_shape = (6, 6, 6)

    cropped = crop_nifti_image_if_necessary(image, target_shape)

    assert cropped.shape == target_shape
    assert np.count_nonzero(cropped.get_fdata()) == 8

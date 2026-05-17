import nibabel as nib
import numpy as np

from pyment.preprocessing.pad import pad_nifti_image_if_necessary


def test_pad_nifti_image_if_necessary_shape_and_edge_mode():
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
    image = nib.Nifti1Image(data, affine=np.eye(4))
    target_shape = (4, 5, 6)

    padded = pad_nifti_image_if_necessary(image, target_shape)

    assert padded.shape == target_shape, (
        'Expected pad_nifti_image_if_necessary to yield images with the given '
        'target_shape'
    )

    padded_data = padded.get_fdata()
    assert np.allclose(padded_data[1:3, 1:4, 1:5], data), (
        'Expected pad_nifti_image_if_necessary to center the original data '
        'in the padded output'
    )
    assert np.allclose(padded_data[0, 0, 0], data[0, 0, 0]), (
        'Expected pad_nifti_image_if_necessary to use edge padding at the start'
    )
    assert np.allclose(padded_data[-1, -1, -1], data[-1, -1, -1]), (
        'Expected pad_nifti_image_if_necessary to use edge padding at the end'
    )


def test_pad_nifti_image_if_necessary_affine_shift():
    data = np.zeros((2, 2, 2), dtype=np.float32)
    affine = np.eye(4)
    affine[:3, 3] = [10.0, 20.0, 30.0]
    image = nib.Nifti1Image(data, affine=affine)
    target_shape = (4, 4, 4)

    padded = pad_nifti_image_if_necessary(image, target_shape)

    # Padding adds 1 voxel on each side for each dimension.
    expected_shift = np.asarray([1.0, 1.0, 1.0])
    expected_affine = affine.copy()
    expected_affine[:3, 3] -= expected_affine[:3, :3] @ expected_shift

    assert np.allclose(padded.affine, expected_affine), (
        'Expected pad_nifti_image_if_necessary to shift the affine to '
        'account for the padding offset'
    )

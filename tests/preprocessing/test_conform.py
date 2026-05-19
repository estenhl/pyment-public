import nibabel as nib
import numpy as np
import pytest
from numpy.typing import DTypeLike

from pyment.preprocessing import conform, rescale

TARGET_SHAPE = (224, 192, 224)


def make_nifti(
    shape: tuple,
    zooms: tuple = (1.0, 1.0, 1.0),
    dtype: DTypeLike = np.float32,
) -> nib.Nifti1Image:
    data = np.ones(shape, dtype=dtype)
    affine = np.diag([*zooms, 1.0])
    image = nib.Nifti1Image(data, affine=affine)
    image.header.set_data_dtype(dtype)
    return image


def test_conform_data_type(nifti_image):
    conformed_image = conform(nifti_image)

    assert conformed_image.header.get_data_dtype() == np.dtype(np.float32), (
        'Expected conform to yield the hardcoded np.float32 header dtype'
    )
    assert np.asarray(conformed_image.dataobj).dtype == np.dtype(np.float32), (
        'Expected conform to yield the hardcoded np.float32 data dtype'
    )


def test_conform_converts_non_float32_data():
    image = make_nifti((100, 100, 100), dtype=np.float64)

    conformed = conform(image)

    assert np.asarray(conformed.dataobj).dtype == np.dtype(np.float32), (
        'Expected conform to convert data dtype to float32'
    )


def test_conform_raises_for_4d_image():
    data = np.ones((10, 10, 10, 2), dtype=np.float32)
    image = nib.Nifti1Image(data, affine=np.eye(4))

    with pytest.raises(NotImplementedError):
        conform(image)


def test_conform_voxel_size(nifti_image):
    rescaled_image = rescale(nifti_image, target_resolution=(1.5, 1.5, 1.5))

    assert np.allclose(rescaled_image.header.get_zooms(), 1.5), (
        'Expected rescale to yield zooms matching the ratios given as a '
        'target_resolution parameter'
    )

    conformed_image = conform(rescaled_image)

    assert np.allclose(conformed_image.header.get_zooms(), 1), (
        'Expected conform to yield unchanged zooms when no target_resolution '
        'is given'
    )


def test_conform_rescales_when_voxels_too_large():
    image = make_nifti((112, 96, 112), zooms=(2.0, 2.0, 2.0))

    conformed = conform(image)

    assert np.allclose(conformed.header.get_zooms()[:3], 1.0, atol=0.05), (
        'Expected conform to rescale to unit voxels when input voxels are '
        'too large'
    )


def test_conform_rescales_when_voxels_too_small():
    image = make_nifti((448, 384, 448), zooms=(0.5, 0.5, 0.5))

    conformed = conform(image)

    assert np.allclose(conformed.header.get_zooms()[:3], 1.0, atol=0.05), (
        'Expected conform to rescale to unit voxels when input voxels are '
        'too small'
    )


def test_conform_does_not_rescale_when_voxels_in_range():
    image = make_nifti(TARGET_SHAPE, zooms=(1.0, 1.0, 1.0))

    conformed = conform(image)

    assert np.allclose(conformed.header.get_zooms()[:3], 1.0), (
        'Expected conform to leave voxel size unchanged when zooms are '
        'within the accepted range'
    )


def test_conform_crops_oversized_image():
    image = make_nifti((300, 300, 300), zooms=(1.0, 1.0, 1.0))

    conformed = conform(image)

    assert conformed.shape == TARGET_SHAPE, (
        'Expected conform to crop an oversized image to the target shape'
    )


def test_conform_pads_undersized_image():
    image = make_nifti((100, 100, 100), zooms=(1.0, 1.0, 1.0))

    conformed = conform(image)

    assert conformed.shape == TARGET_SHAPE, (
        'Expected conform to pad an undersized image to the target shape'
    )


def test_conform_reshape_if_necessary(nifti_image):
    conformed_image = conform(nifti_image)

    assert conformed_image.shape == TARGET_SHAPE, (
        'Expected conform to yield images with the hardcoded image shape of '
        '(224, 192, 224)'
    )


def test_rescale_output_shape():
    image = make_nifti((100, 100, 100), zooms=(2.0, 2.0, 2.0))

    rescaled = rescale(image, target_resolution=(1.0, 1.0, 1.0))

    assert rescaled.shape == (200, 200, 200), (
        'Expected rescale to double the shape when halving the voxel size'
    )


def test_rescale_output_zooms():
    image = make_nifti((100, 100, 100), zooms=(2.0, 2.0, 2.0))

    rescaled = rescale(image, target_resolution=(1.0, 1.0, 1.0))

    assert np.allclose(rescaled.header.get_zooms()[:3], 1.0), (
        'Expected rescale to yield zooms matching target_resolution'
    )


def test_rescale_anisotropic():
    image = make_nifti((100, 100, 100), zooms=(1.0, 2.0, 1.0))

    rescaled = rescale(image, target_resolution=(1.0, 1.0, 1.0))

    assert rescaled.shape == (100, 200, 100), (
        'Expected rescale to scale only the axis whose zoom differs from '
        'target_resolution'
    )

import numpy as np

from pyment.preprocessing import conform, rescale


def test_conform_data_type(nifti_image):
    conformed_image = conform(nifti_image)

    assert conformed_image.header.get_data_dtype() == np.dtype(np.float32), (
        'Expected conform to yield the harcoded np.float32 header dtype'
    )
    assert conformed_image.dataobj.dtype == np.dtype(np.float32), (
        'Expected conform to yield the hardcoded np.float32 data dtype'
    )


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


def test_conform_reshape_if_necessary(nifti_image):
    conformed_image = conform(nifti_image)

    assert conformed_image.shape == (224, 192, 224), (
        'Expected conform to yield images with the hardcoded image shape of '
        '(224, 192, 224)'
    )

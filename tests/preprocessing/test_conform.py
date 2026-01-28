import nibabel as nib
import numpy as np

from pyment.preprocessing import conform, rescale


def test_conform_data_type(nifti_image):
    conformed_image = conform(nifti_image)

    assert conformed_image.header.get_data_dtype() == np.dtype(np.float32), (
        'Conforming image does not change the dtype of the image as reported '
        'in the image header'
    )
    assert conformed_image.dataobj.dtype == np.dtype(np.float32), (
        'Conforming image does not change the dtype of the data array '
        'contained in the nifti image'
    )

def test_conform_voxel_size(nifti_image):
    rescaled_image = rescale(nifti_image, target_resolution=(1.5, 1.5, 1.5))

    assert np.allclose(rescaled_image.header.get_zooms(), 1.5), (
        'Rescaling image does not yield the correct resolution'
    )

    conformed_image = conform(rescaled_image)

    assert np.allclose(conformed_image.header.get_zooms(), 1), (
        'Conforming image does not set the voxel size to 1x1x1'
    )

def test_conform_reshape_if_necessary(nifti_image):
    conformed_image = conform(nifti_image)

    assert conformed_image.shape == (224, 192, 224), (
        'Conforming image does not yield the expected image size of '
        '224x192x224'
    )
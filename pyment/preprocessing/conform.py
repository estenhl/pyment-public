import logging
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .crop import crop_nifti_image_if_necessary
from .pad import pad_nifti_image_if_necessary


logger = logging.getLogger(__name__)

def rescale(
    image: nib.Nifti1Image,
    target_resolution = np.asarray([1.0, 1.0, 1.0]),
    order: int = 1
):
    current_shape = np.asarray(image.shape[:3])
    current_resolution = np.asarray(image.header.get_zooms()[:3])

    scale = current_resolution / target_resolution
    new_shape = tuple(np.round(current_shape * scale).astype(int))

    target_affine = image.affine.copy()
    target_affine[:3, :3] *= target_resolution / current_resolution

    resampled = resample_from_to(
        image,
        (new_shape, target_affine),
        order=order
    )

    return resampled

def conform(
    image: nib.Nifti1Image,
    target_shape: tuple[int, int, int] = np.asarray([224, 192, 224]),
    target_dtype: np.dtype = np.float32
) -> nib.Nifti1Image:
    """Conforms an image to the expected format, i.e. a 3-dimensional
    image with the given shape (defaults to 224x192x224). If a
    redundant channel-dimension exists, it is removed. If the image is
    currently too large along any dimension, a "central" crop is made
    by determining the bound of the brain (e.g. non-zero voxels) and
    retaining equivalent padding oneach side. If the image is currently
    too small along either axis, the image is zero-padded equally on
    each side.

    Parameters
    ----------
    image : np.ndarray
        A three-dimensional or four-dimensional tensor containing raw
        voxel-values.

    Returns
    -------
    np.ndarray
        The conformed image.
    """
    if len(image.dataobj.shape) > 3:
        raise NotImplementedError(
            'Conforming images with a channel-dimension is not implemented'
        )

    if image.header.get_data_dtype() != np.dtype(target_dtype):
        image.header.set_data_dtype(target_dtype)
        logger.debug('Sat header dtype to %s', target_dtype)

    if image.dataobj.dtype != np.dtype(target_dtype):
        image = nib.Nifti1Image(
            image.get_fdata(dtype=target_dtype),
            header=image.header,
            affine=image.affine
        )
        logger.debug('Sat data dtype to %s', target_dtype)

    if (
        np.amax(image.header.get_zooms()[:3]) > 1.03 or
        np.amin(image.header.get_zooms()[:3]) < 0.69
    ):
        shape_before = image.shape
        scale_before = image.header.get_zooms()
        image = rescale(image)
        logger.warning(
            (
                'Voxel-size %s outside the range of training data. Rescaled '
                'image to %s with shape %s (from original shape %s)'
            ),
            scale_before,
            image.header.get_zooms(),
            image.shape,
            shape_before
        )

    if np.any(image.shape > target_shape):
        image = crop_nifti_image_if_necessary(image, target_shape=target_shape)
        logger.debug('Cropped image to shape %s', image.shape)

    if np.any(image.shape < target_shape):
        image = pad_nifti_image_if_necessary(image, target_shape=target_shape)
        logger.debug('Padded image to shape %s', image.shape)

    return image

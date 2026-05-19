"""Utilities for conforming NIfTI images to the shape and voxel size
expected by the SFCN model."""

import logging

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from numpy.typing import DTypeLike

from .crop import crop_nifti_image_if_necessary
from .pad import pad_nifti_image_if_necessary

logger = logging.getLogger(__name__)


def rescale(
    image: nib.Nifti1Image,
    target_resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 1,
) -> nib.Nifti1Image:
    """Resamples image to target_resolution using spline interpolation.

    Parameters
    ----------
    image : nib.Nifti1Image
        The image to resample.
    target_resolution : tuple[float, float, float]
        Target voxel size in mm for each axis.
    order : int
        Spline interpolation order passed to nibabel (0=nearest,
        1=linear).

    Returns
    -------
    nib.Nifti1Image
        Resampled image with voxel size matching target_resolution.
    """
    current_shape = np.asarray(image.shape[:3])
    current_resolution = np.asarray(image.header.get_zooms()[:3])

    scale = current_resolution / np.asarray(target_resolution)
    new_shape = tuple(np.round(current_shape * scale).astype(int))

    target_affine = (
        image.affine.copy() if image.affine is not None else np.eye(4)
    )
    target_affine[:3, :3] *= np.asarray(target_resolution) / current_resolution

    resampled = resample_from_to(image, (new_shape, target_affine), order=order)

    return resampled


def conform(
    image: nib.Nifti1Image,
    target_shape: tuple[int, int, int] = (224, 192, 224),
    target_dtype: DTypeLike = np.float32,
) -> nib.Nifti1Image:
    """Conforms image to target_shape and target_dtype.

    Voxels outside the range [0.69, 1.03] mm are rescaled to 1 mm
    isotropic. Images larger than target_shape are centrally cropped
    to the bounding box of non-zero voxels. Images smaller than
    target_shape are edge-padded.

    Parameters
    ----------
    image : nib.Nifti1Image
        A 3-D image. 4-D inputs raise NotImplementedError.
    target_shape : tuple[int, int, int]
        Desired output shape.
    target_dtype : DTypeLike
        Desired output dtype.

    Returns
    -------
    nib.Nifti1Image
        Conformed image with shape target_shape and dtype target_dtype.

    Raises
    ------
    NotImplementedError
        If image has more than 3 dimensions.
    """
    if len(image.dataobj.shape) > 3:
        raise NotImplementedError(
            'Conforming images with a channel-dimension is not implemented'
        )

    if image.header.get_data_dtype() != np.dtype(target_dtype):
        image.header.set_data_dtype(target_dtype)
        logger.debug('Sat header dtype to %s', target_dtype)

    if np.asarray(image.dataobj).dtype != np.dtype(target_dtype):
        image = nib.Nifti1Image(
            image.get_fdata(dtype=target_dtype),
            header=image.header,
            affine=image.affine,
        )
        logger.debug('Sat data dtype to %s', target_dtype)

    if (
        np.amax(image.header.get_zooms()[:3]) > 1.03
        or np.amin(image.header.get_zooms()[:3]) < 0.69
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
            shape_before,
        )

    if np.any(image.shape > target_shape):
        image = crop_nifti_image_if_necessary(image, target_shape=target_shape)
        logger.debug('Cropped image to shape %s', image.shape)

    if np.any(image.shape < target_shape):
        image = pad_nifti_image_if_necessary(image, target_shape=target_shape)
        logger.debug('Padded image to shape %s', image.shape)

    return image

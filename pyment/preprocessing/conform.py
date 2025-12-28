import logging
import nibabel as nib
import numpy as np
from typing import Tuple, Union

from .crop import crop_numpy_array_if_necessary


logger = logging.getLogger(__name__)

def _pad_if_necessary(
    image: np.ndarray,
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    pad = [(0, 0)] * 3

    for dim in range(3):
        if image.shape[dim] < target_shape[dim]:
            padding = target_shape[dim] - image.shape[dim]
            first_padding = padding // 2
            second_padding = padding - first_padding
            pad[dim] = (first_padding, second_padding)

    return np.pad(image, tuple(pad), mode='constant', constant_values=0)

def _center_crop_or_pad(
    image: np.ndarray,
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    image = _pad_if_necessary(image, target_shape)
    image = crop_numpy_array_if_necessary(image, target_shape)

    return image

def conform(image: np.ndarray) -> np.ndarray:
    """Conforms an image to the expected format if necessary. The
    expected format means an image of shape 224x192x224. If the image
    has a redundant channel-dimension, this is removed. If the image is
    currently too large along any dimension, a "central" crop is made
    by determining the bound of the brain (e.g. non-zero voxels) and
    retaining equivalent padding on each side. If the image is
    currently too small along either axis, the image is zero-padded
    equally on each side.

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
    logger.debug('Original image shape: %s', str(image.shape))

    image = to_float32(image)

    if len(image.shape) == 4:
        if image.shape[-1] != 1:
            raise ValueError(f'Unable to handle multi-channel images')

        image = image[...,0]

    if image.shape != (224, 192, 224):
        image = _center_crop_or_pad(image, (224, 192, 224))

    logger.debug('Conformed image shape: %s', str(image.shape))

    return image

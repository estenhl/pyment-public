import logging
import numpy as np
from typing import Tuple


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

def _crop_if_necessary(
    image: np.ndarray,
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    nonzero = np.where(image != 0)

    idx = [slice(None)] * 3

    for dim in np.arange(3):
        if image.shape[dim] > target_shape[dim]:
            extrafluous = target_shape[dim] / 2
            center = np.round(np.mean([
                np.amin(nonzero[dim]),
                np.amax(nonzero[dim])
            ]))
            min_idx = int(center - extrafluous)
            max_idx = int(center + extrafluous)

            if min_idx < 0:
                max_idx -= min_idx
                min_idx = 0

            if max_idx > image.shape[dim]:
                diff = max_idx - image.shape[dim]
                min_idx -= diff
                max_idx -= diff

            idx[dim] = slice(min_idx, max_idx)

    image = image[tuple(idx)]

    return image

def _center_crop_or_pad(
    image: np.ndarray,
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    image = _pad_if_necessary(image, target_shape)
    image = _crop_if_necessary(image, target_shape)

    return image

def conform(
    image: np.ndarray,
    relative_normalization: bool = False
) -> np.ndarray:
    """Conforms an image to the expected format if necessary. The
    expected format means an image of shape 224x192x224 with voxel
    values spanning the range [0, 1]. If the image has a redundant
    channel-dimension, this is removed. If the image is currently too
    large along any dimension, a "central" crop is made by determining
    the bound of the brain (e.g. non-zero voxels) and retaining
    equivalent padding on each side. If the image is currently too small
    along either axis, the image is zero-padded equally on each side.
    If the voxel-values does not fall within the expected range, they
    are normalized. If the relative_normalization-flag is set, the
    values are normalized by dividing by the image max, otherwise they
    are divided by 255. However, if the largest value is >255, this
    indicates that the image has not been processed with FastSurfer,
    and an error is raised.

    Parameters
    ----------
    image : np.ndarray
        A three-dimensional or four-dimensional tensor containing raw
        voxel-values.
    relative_normalization : bool
        If set, the voxel values are normalized by dividing by the image
        max, otherwise they are divided by 255.

    Returns
    -------
    np.ndarray
        The conformed image.
    """
    logger.debug('Original image shape: %s', str(image.shape))

    image = image.astype(np.float32)

    if len(image.shape) == 4:
        if image.shape[-1] != 1:
            raise ValueError(f'Unable to handle multi-channel images')

        image = image[...,0]

    if image.shape != (224, 192, 224):
        image = _center_crop_or_pad(image, (224, 192, 224))

    logger.debug('Conformed image shape: %s', str(image.shape))
    logger.debug(
        'Original image voxel value range: %f-%f',
        np.amin(image), np.amax(image)
    )

    if relative_normalization:
        image -= np.amin(image)
        image /= np.amax(image)
        image *= 255.0

    logger.debug(
        'Conformed image voxel value range: %f-%f',
        np.amin(image), np.amax(image)
    )

    return image

"""Utilities for cropping NIfTI images to a target shape."""

import nibabel as nib
import numpy as np


def calculate_bounds(
    image: np.ndarray, target_shape: tuple[int, int, int]
) -> list:
    """Computes per-axis slice indices to crop image to target_shape,
    centred on the bounding box of non-zero voxels.

    Parameters
    ----------
    image : np.ndarray
        3-D array of voxel values.
    target_shape : tuple[int, int, int]
        Desired output size per axis.

    Returns
    -------
    list[slice]
        One slice per axis; axes that do not require cropping keep
        slice(None).
    """
    nonzero = np.where(image != 0)
    idx = [slice(None)] * 3

    for dim in np.arange(3):
        if image.shape[dim] > target_shape[dim]:
            extrafluous = target_shape[dim] / 2
            center = np.round(
                np.mean([np.amin(nonzero[dim]), np.amax(nonzero[dim])])
            )
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

    return idx


def crop_nifti_image_if_necessary(
    image: nib.Nifti1Image, target_shape: tuple[int, int, int]
) -> nib.Nifti1Image:
    """Crops image to target_shape along any axis that exceeds it,
    centred on the bounding box of non-zero voxels.

    Parameters
    ----------
    image : nib.Nifti1Image
        Image to crop.
    target_shape : tuple[int, int, int]
        Maximum size per axis.

    Returns
    -------
    nib.Nifti1Image
        Image with each axis <= the corresponding target dimension.
    """
    idx = calculate_bounds(image.get_fdata(), target_shape)

    sliced = image.slicer[tuple(idx)]

    return sliced

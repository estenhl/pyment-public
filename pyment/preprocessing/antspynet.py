"""Utilities for preprocessing images with antspynet skullstripping."""

import ants
import antspynet
import numpy as np


def rescale_intensity(
    image: ants.ANTsImage,
    dst_min: float = 0.0,
    dst_max: float = 255.0,
    f_low: float = 0.0,
    f_high: float = 0.999,
) -> ants.ANTsImage:
    """Rescales voxel intensities to the range [dst_min, dst_max].

    Estimates a source floor and ceiling as the f_low and f_high
    percentiles of image, linearly scales that range onto
    [dst_min, dst_max], and clips values that fall outside of it.
    This mirrors the robust intensity scaling FreeSurfer/FastSurfer
    apply when conforming a raw scan to their 8-bit intensity range,
    which the SFCN model expects: raw scanner intensities are
    arbitrary and vary per scan, so a fixed source range can't be
    hardcoded, hence the per-image percentile estimate.

    Parameters
    ----------
    image : ants.ANTsImage
        The image to rescale.
    dst_min : float
        Output value the f_low percentile is mapped to.
    dst_max : float
        Output value the f_high percentile is mapped to.
    f_low : float
        Lower percentile (in [0, 1]) used to estimate the source
        floor.
    f_high : float
        Upper percentile (in [0, 1]) used to estimate the source
        ceiling.

    Returns
    -------
    ants.ANTsImage
        Image with intensities linearly rescaled and clipped to
        [dst_min, dst_max].
    """
    data = image.numpy()
    src_min = np.percentile(data, f_low * 100)
    src_max = np.percentile(data, f_high * 100)

    if src_max > src_min:
        scale = (dst_max - dst_min) / (src_max - src_min)
        rescaled = (data - src_min) * scale + dst_min
    else:
        # No contrast to stretch (e.g. a constant image); avoid a
        # division by zero.
        rescaled = np.full_like(data, dst_min)

    rescaled = np.clip(rescaled, dst_min, dst_max)

    return image.new_image_like(rescaled)


def skullstrip_with_antspynet(
    image: ants.ANTsImage,
    orientation: str = 'RSP',
    target_resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    modality: str = 't1',
) -> ants.ANTsImage:
    """Reorients, resamples and skullstrips image using antspynet.

    Reorients image to orientation, resamples to target_resolution
    isotropic voxels, rescales intensities to match FreeSurfer's
    conformed intensity range (see rescale_intensity), runs antspynet
    brain extraction, masks out non-brain voxels, and crops to the
    bounding box of the brain mask.

    Parameters
    ----------
    image : ants.ANTsImage
        The image to preprocess.
    orientation : str
        Target ANTs orientation code.
    target_resolution : tuple[float, float, float]
        Target voxel size in mm for each axis.
    modality : str
        Modality passed to antspynet.brain_extraction.

    Returns
    -------
    ants.ANTsImage
        Skullstripped image cropped to the brain mask bounding box.
    """
    image = ants.reorient_image2(image, orientation=orientation)

    image = ants.resample_image(
        image, target_resolution, use_voxels=False, interp_type=4
    )

    image = rescale_intensity(image)

    mask_probability = antspynet.brain_extraction(image, modality=modality)
    mask = ants.threshold_image(mask_probability, 0.5, 1.0, 1, 0)
    brain = image * mask

    brain = ants.crop_image(brain, mask)

    return brain

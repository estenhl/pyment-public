"""Utilities for lazy generation of FastSurfer crop volumes."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
from nibabel.spatialimages import SpatialImage
from tqdm import tqdm

from pyment.preprocessing import conform

logger = logging.getLogger(__name__)


def ensure_fastsurfer_crop_exists(folder: str) -> bool:
    """Ensure a crop volume exists for a single subject folder.

    Generates ``mri/crop.mgz`` from ``mri/orig.mgz`` masked by
    ``mri/mask.mgz`` if it does not already exist. Returns
    ``False`` if either source file is missing.

    Parameters
    ----------
    folder : str
        Path to the FastSurfer subject folder.

    Returns
    -------
    bool
        ``True`` if the crop exists or was successfully created,
        ``False`` if a source file was missing.
    """

    crop_path = os.path.join(folder, 'mri', 'crop.mgz')

    if not os.path.exists(crop_path):
        logger.debug('Creating crop for %s', folder)
        mask = os.path.join(folder, 'mri', 'mask.mgz')
        orig = os.path.join(folder, 'mri', 'orig.mgz')

        if not os.path.isfile(orig):
            logger.debug('Unable to create crop for %s: Missing orig', folder)

            return False
        elif not os.path.isfile(mask):
            logger.debug('Unable to create crop for %s: Missing mask', folder)

            return False

        orig = nib.load(orig)
        mask = nib.load(mask)
        assert isinstance(orig, SpatialImage)
        assert isinstance(mask, SpatialImage)

        crop = nib.Nifti1Image(
            orig.get_fdata() * mask.get_fdata(),
            affine=orig.affine,
            header=orig.header,
        )

        crop = conform(crop)

        logger.debug('Writing crop to %s', crop_path)
        nib.save(crop, crop_path)

    return True


def ensure_fastsurfer_crops_exists(
    folders: list[str], num_threads: int = 1
) -> bool:
    """Ensure crops exist for multiple subject folders in parallel.

    Calls ``ensure_fastsurfer_crop_exists`` for each folder using
    a thread pool. Logs a warning for failures but does not raise.

    Parameters
    ----------
    folders : list[str]
        Paths to FastSurfer subject folders.
    num_threads : int, optional
        Maximum number of threads for parallel processing.

    Returns
    -------
    bool
        ``True`` if all crops were created successfully.
    """

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda folder: ensure_fastsurfer_crop_exists(folder),
                    folders,
                ),
                total=len(folders),
            )
        )

    if not all(results):
        logger.warning(
            'Failed to create crops for %d folders', len(results) - sum(results)
        )

    return all(results)

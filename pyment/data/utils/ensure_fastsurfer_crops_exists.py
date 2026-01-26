import logging
import os
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from nibabel.processing import resample_from_to

from pyment.preprocessing import conform


logger = logging.getLogger(__name__)

def ensure_fastsurfer_crop_exists(
    folder: str,
    target_shape: tuple[int, int, int]
) -> bool:
    crop_path = os.path.join(folder, 'mri', 'crop.mgz')

    if not os.path.exists(crop_path):
        logger.debug('Creating crop for %s', folder)
        mask = os.path.join(folder, 'mri', 'mask.mgz')
        orig = os.path.join(folder, 'mri', 'orig.mgz')

        if not os.path.isfile(orig):
            logger.debug(
                'Unable to create crop for %s: Missing orig', folder
            )
            return False
        elif not os.path.isfile(mask):
            logger.debug(
                'Unable to create crop for %s: Missing mask', folder
            )
            return False

        orig = nib.load(orig)
        orig_data = orig.get_fdata()
        # orig_data -= np.amin(orig_data)
        # orig_data /= np.amax(orig_data)
        # orig_data *= 255.0

        mask = nib.load(mask)

        crop = nib.Nifti1Image(
            orig_data * mask.get_fdata(),
            affine=orig.affine,
            header=orig.header
        )

        crop = conform(crop)

        logger.debug('Writing crop to %s', crop_path)
        nib.save(crop, crop_path)

    return True


def ensure_fastsurfer_crops_exists(
    folders: list[str],
    target_shape: tuple[int, int, int],
    num_threads: int = 1
):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(
            executor.map(
                lambda folder: ensure_fastsurfer_crop_exists(
                    folder,
                    target_shape
                ),
                folders
            )
        )

    if not all(results):
        logger.warning(
            'Failed to create crops for %d folders', 
            len(results) - sum(results)
        )

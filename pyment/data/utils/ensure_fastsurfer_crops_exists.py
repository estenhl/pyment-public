import logging
import os
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor

from pyment.preprocessing import crop_nifti_image_if_necessary


logger = logging.getLogger(__name__)

def ensure_fastsurfer_crop_exists(
    folder: str,
    target_shape: tuple[int, int, int]
) -> bool:
    crop_path = os.path.join(folder, 'mri', 'crop.mgz')

    if not os.path.exists(crop_path):
        logger.debug('Creating crop for %s', folder)
        mask = nib.load(os.path.join(folder, 'mri', 'mask.mgz'))
        orig = nib.load(os.path.join(folder, 'mri', 'orig.mgz'))

        crop = nib.Nifti1Image(
            mask.get_fdata() * orig.get_fdata(),
            affine=orig.affine,
            header=orig.header
        )
        crop = crop_nifti_image_if_necessary(crop, target_shape)

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
        raise ValueError('Failed to create crops for some folders')

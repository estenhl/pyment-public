import nibabel as nib
import numpy as np

def calculate_bounds(
    image: np.ndarray,
    target_shape: tuple[int, int, int]
):
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

    return idx

def crop_nifti_image_if_necessary(
    image: nib.Nifti1Image,
    target_shape: tuple[int, int, int]
) -> nib.Nifti1Image:
    idx = calculate_bounds(image.get_fdata(), target_shape)

    sliced = image.slicer[tuple(idx)]

    return sliced
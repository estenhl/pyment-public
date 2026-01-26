import nibabel as nib
import numpy as np


def pad_nifti_image_if_necessary(
    image: nib.Nifti1Image,
    target_shape: tuple[int, int, int]
) -> nib.Nifti1Image:
    pad = [(0, 0)] * 3

    for dim in range(3):
        if image.shape[dim] < target_shape[dim]:
            padding = target_shape[dim] - image.shape[dim]
            first_padding = padding // 2
            second_padding = padding - first_padding
            pad[dim] = (first_padding, second_padding)

    data = np.pad(image.dataobj, tuple(pad), mode='edge')
    affine = image.affine.copy()
    affine[:3, 3] -= affine[:3, :3] @ np.asarray([p[0] for p in pad])

    image = nib.Nifti1Image(data, header=image.header, affine=affine)

    return image

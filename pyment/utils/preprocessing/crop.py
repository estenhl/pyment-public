import os
import nibabel as nib

from typing import Tuple


def crop_mri(src: str, dest: str, bounds: Tuple[Tuple[int]]) -> None:
    """Crops an MRI by the given bounds and stores the result 

    Args:
        src (str): Path to original MRI
        dest (str): Where to store the result
        bounds (Tuple[Tuple[int]]): Bounds to crop by. Contains three 
            pairs, where the first refers to the y-axis, the second x,
            and the third z. Start is inclusive, end is exclusive

    """
    img = nib.load(src)
    data = img.get_fdata()

    if bounds[0][0] < 0:
        raise ValueError('ymin < 0')
    elif bounds[0][1] > data.shape[0]:
        raise ValueError('ymax > data.ymax')
    if bounds[1][0] < 0:
        raise ValueError('xmin < 0')
    elif bounds[1][1] > data.shape[1]:
        raise ValueError('xmax > data.xmax')
    if bounds[2][0] < 0:
        raise ValueError('zmin < 0')
    elif bounds[2][1] > data.shape[2]:
        raise ValueError('zmax > data.zmax')


    data = data[
        bounds[0][0]:bounds[0][1],
        bounds[1][0]:bounds[1][1],
        bounds[2][0]:bounds[2][1],
    ]

    img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    nib.save(img, dest)


def crop_folder(src: str, dest: str, bounds: Tuple[Tuple[int]]) -> None:
    """Crops all MRIs in a folder by the given bounds"""
    for filename in os.listdir(src):
        crop_mri(os.path.join(src, filename), os.path.join(dest, filename),
                 bounds)
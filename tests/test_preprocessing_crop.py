import os
import nibabel as nib
import numpy as np

from pyment.utils.preprocessing import crop_mri


def test_crop_mri_creates_file():
    src = 'src.nii.gz'
    dest = 'dest.nii.gz'

    try:
        img = nib.Nifti1Image(np.ones((10, 10, 10)), affine=np.eye(4))
        borders = ((1, 1), (1, 1), (1, 1))

        nib.save(img, src)
        crop_mri(src, dest, borders)

        assert os.path.isfile(dest), 'crop_mri does not create a cropped file'
    finally:
        if os.path.isfile(src):
            os.remove(src)

        if os.path.isfile(dest):
            os.remove(dest)


def test_crop_mri_crops():
    src = 'src.nii.gz'
    dest = 'dest.nii.gz'

    try:
        data = np.reshape(np.arange(10*10*10), (10, 10, 10))
        img = nib.Nifti1Image(data, affine=np.eye(4))
        borders = ((1, 9), (1, 8), (1, 7))

        nib.save(img, src)
        crop_mri(src, dest, borders)

        img = nib.load(dest)

        assert (8, 7, 6) == img.get_fdata().shape, \
               'crop_mri does not crop correct borders'
        assert np.array_equal(img.get_fdata(), data[1:9,1:8,1:7]), \
               'crop_mri does not crop the correct regions'
    finally:
        if os.path.isfile(src):
            os.remove(src)

        if os.path.isfile(dest):
            os.remove(dest)


def test_crop_mri_file_not_found():
    src = 'src.nii.gz'
    dest = 'dest.nii.gz'

    try:
        borders = ((1, 1), (1, 1), (1, 1))
        crop_mri(src, dest, borders)

        assert False, 'crop_mri with non-existing file does not raise an error'
    except FileNotFoundError:
        pass
    finally:
        if os.path.isfile(src):
            os.remove(src)

        if os.path.isfile(dest):
            os.remove(dest)


def test_crop_mri_invalid_bounds():
    src = 'src.nii.gz'
    dest = 'dest.nii.gz'

    try:
        data = np.reshape(np.arange(10*10*10), (10, 10, 10))
        img = nib.Nifti1Image(data, affine=np.eye(4))
        borders = ((1, 13), (1, 8), (1, 7))

        nib.save(img, src)
        crop_mri(src, dest, borders)

        assert False, 'crop_mri with invalid bounds does not raise an error'
    except ValueError:
        pass
    finally:
        if os.path.isfile(src):
            os.remove(src)

        if os.path.isfile(dest):
            os.remove(dest)
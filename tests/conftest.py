import nibabel as nib
import os

from pytest import fixture


test_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(test_path, os.pardir, 'data')

@fixture
def nifti_image():
    return nib.load(os.path.join(data_path, 'esten.nii.gz'))
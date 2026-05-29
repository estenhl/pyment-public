import os

import nibabel as nib
from pytest import fixture

test_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(test_path, 'fixtures')


@fixture
def nifti_image():
    return nib.load(os.path.join(data_path, 'raw', 'esten.nii.gz'))

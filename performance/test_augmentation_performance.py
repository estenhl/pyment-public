import os

from pytest import fixture


DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
IMAGE_PATH = os.path.join(DATA_FOLDER, 'sample.nii.gz')


@fixture
def dataset(batch_size=12):
    assert os.path.isfile(IMAGE_PATH), \
        ('Unable to perform performance tests without a demo MRI located at '
         f'{IMAGE_PATH}')

def test_zoom(dataset):
    pass

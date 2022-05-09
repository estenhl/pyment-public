import os

from pytest import fixture
from time import time

from pyment.data import NiftiDataset
from pyment.data.augmenters import NiftiAugmenter
from pyment.data.generators import AsyncNiftiGenerator


DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
IMAGE_PATH = os.path.join(DATA_FOLDER, 'orig.nii.gz')

BATCH_SIZE = 14
NUM_THREADS = 8
RUNS = 10


@fixture
def dataset():
    assert os.path.isfile(IMAGE_PATH), \
        ('Unable to perform performance tests without a demo MRI located at '
         f'{IMAGE_PATH}')

    paths = [IMAGE_PATH] * BATCH_SIZE

    dataset = NiftiDataset(paths, labels=None)

    return dataset

def test_zoom(dataset):
    augmenter = NiftiAugmenter(zoom_ranges=[0.05, 0.05, 0.05])
    generator = AsyncNiftiGenerator(dataset,
                                    augmenter=augmenter,
                                    batch_size=BATCH_SIZE,
                                    threads=NUM_THREADS)
    generator = iter(generator)

    for _ in range(10):
        generator.reset()
        start = time()
        for X, y in generator:
            break
        print(f'Zoom: {round(time() - start, 2)} seconds')

def test_complex(dataset):
    augmenter = NiftiAugmenter(
        flip_probabilities = [0.5, 0, 0],
        shift_ranges = [5, 5, 5],
        zoom_ranges = [0.1, 0.1, 0.1],
        rotation_ranges = [5, 5, 5],
        shear_ranges = [0.05, 0.05, 0.05]
    )
    generator = AsyncNiftiGenerator(dataset,
                                    augmenter=augmenter,
                                    batch_size=BATCH_SIZE,
                                    threads=NUM_THREADS)
    generator = iter(generator)

    for _ in range(10):
        generator.reset()
        start = time()
        for X, y in generator:
            break
        print(f'Complex: {round(time() - start, 2)} seconds')

import argparse
import logging
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from time import time

from utils import configure_environment

configure_environment()

from pyment.data.augmenters import NiftiAugmenter


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)


def _pad(image: np.ndarray, size: int) -> np.ndarray:
    padding = size - np.asarray(image.shape)
    before = padding // 2
    after = padding - before
    padding = list(zip(before, after))

    return np.pad(image, padding)


def augment_nifti(augmenter: str, image: str, iterations: int = 1,
                  destination: str = None):
    np.random.seed(42)

    augmenter = NiftiAugmenter.from_file(augmenter)

    image = nib.load(image).get_fdata()
    start = time()
    augmentations = [augmenter(image) for _ in range(iterations)]
    logger.info((f'Performed {iterations} augmentations in {time() - start} '
                 'seconds'))


    max_dim = np.amax(image.shape)
    canvas = np.zeros((max_dim * (iterations + 1), max_dim * 3))

    image = _pad(image, max_dim)
    augmentations = [_pad(augmented, max_dim) for augmented in augmentations]
    means = np.asarray(image.shape) // 2

    for i in range(len(means)):
        slice = np.take(image, means[i], axis=i)
        canvas[0:max_dim,max_dim*i:max_dim*(i+1)] = slice

        for j in range(len(augmentations)):
            slice = np.take(augmentations[j], means[i], axis=i)
            canvas[max_dim*(j+1):max_dim*(j+2),max_dim*i:max_dim*(i+1)] = slice

    plt.figure(figsize=(5, 20))
    plt.imshow(canvas, cmap='Greys_r')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Applies a NiftiAugmenter to a given '
                                      'nifti image a given amount of times'))

    parser.add_argument('-a', '--augmenter', required=True,
                        help='Path to augmenter')
    parser.add_argument('-i', '--image', required=True,
                        help='Path to image')
    parser.add_argument('-n', '--iterations', required=False, default=1,
                        type=int,
                        help='Number of times to apply the augmenter')

    args = parser.parse_args()

    augment_nifti(augmenter=args.augmenter, image=args.image,
                  iterations=args.iterations)

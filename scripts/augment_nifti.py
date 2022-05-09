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

def augment_nifti(augmenter: str, image: str, iterations: int = 1):
    np.random.seed(42)

    augmenter = NiftiAugmenter.from_file(augmenter)

    print(augmenter)

    image = nib.load(image).get_fdata()
    start = time()
    augmentations = [augmenter(image) for _ in range(iterations)]
    logger.info((f'Performed {iterations} augmentations in {time() - start} '
                 'seconds'))

    means = np.asarray(image.shape) // 2

    fig, ax = plt.subplots(iterations + 1, len(means), figsize=(15, 10*(iterations + 1)))
    for i in range(len(means)):
        ax[0][i].imshow(np.take(image, means[i], axis=i), cmap='Greys_r')
        ax[0][i].axis('off')

        for j in range(len(augmentations)):
            ax[1+j][i].imshow(np.take(augmentations[j], means[i], axis=i),
                              cmap='Greys_r')
            ax[1+j][i].axis('off')

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

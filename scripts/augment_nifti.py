import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from utils import configure_environment

configure_environment()

from pyment.data.augmenters import NiftiAugmenter


def augment_nifti(augmenter: str, image: str, iterations: int = 1):
    augmenter = NiftiAugmenter.from_file(augmenter)
    image = nib.load(image).get_fdata()

    idx = np.asarray(image.shape) // 2
    print(idx)

    fig, ax = plt.subplots(iterations + 1, 3)
    ax[0][0].imshow()

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

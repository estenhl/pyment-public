"""Configures a NiftiAugmenter

Example usage:
    python scripts/configure_nifti_augmenter.py \
        --flip_probabilities 0.5 0.0 0.0 \
        --destination /path/to/destination.json
"""


import argparse

from typing import List

from utils import configure_environment

configure_environment()

from pyment.data.augmenters import NiftiAugmenter


def configure_nifti_augmenter(*, flip_probabilities: List[float] = None,
                              destination: str) -> NiftiAugmenter:
    augmenter = NiftiAugmenter(flip_probabilities=flip_probabilities)

    augmenter.save(destination)

    return augmenter


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Configures and saves a NiftiAugmenter')

    parser.add_argument('-f', '--flip_probabilities', required=False,
                        default=None, nargs='+', type=float,
                        help=('Flip probabilities used by the augmenter. If '
                              'used, should be a list of three floats, each '
                              'describing the probability of performing a '
                              'flip along the given axis'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where augmenter is stored')

    args = parser.parse_args()

    configure_nifti_augmenter(flip_probabilities=args.flip_probabilities,
                              destination=args.destination)

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
                              shift_ranges: List[int] = None,
                              zoom_ranges: List[float] = None,
                              rotation_ranges: List[int] = None,
                              shear_ranges: List[float] = None,
                              destination: str) -> NiftiAugmenter:
    augmenter = NiftiAugmenter(flip_probabilities=flip_probabilities,
                               shift_ranges=shift_ranges,
                               zoom_ranges=zoom_ranges,
                               rotation_ranges=rotation_ranges,
                               shear_ranges=shear_ranges)

    augmenter.save(destination)

    return augmenter


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Configures and saves a NiftiAugmenter')

    parser.add_argument('-f', '--flip_probabilities', required=False,
                        default=None, nargs=3, type=float,
                        help=('Flip probabilities used by the augmenter. If '
                              'used, should be a list of three floats, each '
                              'describing the probability of performing a '
                              'flip along the given axis'))
    parser.add_argument('-s', '--shift_ranges', required=False, default=None,
                        nargs=3, type=int,
                        help=('Shift ranges used by the augmenter. If used, '
                              'should be a list of three integers, each '
                              'describing the maximal allowed shift in '
                              'negative and positive direction of the given '
                              'axis'))
    parser.add_argument('-z', '--zoom_ranges', required=False, default=None,
                        nargs=3, type=float,
                        help=('Zoom ranges used by the augmenter. If used, '
                              'should be a list of three floats, each '
                              'describing the minimum and maximum allowed '
                              'zoom along the given axis'))
    parser.add_argument('-r', '--rotation_ranges', required=False,
                        default=None, nargs=3, type=int,
                        help=('Rotation ranges used by the augmenter. If '
                              'used, should be a list of three integers, each '
                              'describing the maximal rotation in both '
                              'directions for the given axis'))
    parser.add_argument('-e', '--shear_ranges', required=False, default=None,
                        type=float, nargs='+',
                        help=('Shear ranges used by the augmenter. If used, '
                              'should be a list of three floats, each '
                              'describing the fraction of shearing allwed '
                              'both in the positive and negativ direction '
                              'of the given axis'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where augmenter is stored')

    args = parser.parse_args()

    configure_nifti_augmenter(flip_probabilities=args.flip_probabilities,
                              shift_ranges=args.shift_ranges,
                              zoom_ranges=args.zoom_ranges,
                              rotation_ranges=args.rotation_ranges,
                              shear_ranges=args.shear_ranges,
                              destination=args.destination)

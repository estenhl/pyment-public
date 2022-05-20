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
                              noise_threshold: float = None,
                              intensity_threshold: float = None,
                              blur_threshold: float = None,
                              blur_probability: float = None,
                              crop_box_sides: int = None,
                              destination: str) -> NiftiAugmenter:
    augmenter = NiftiAugmenter(flip_probabilities=flip_probabilities,
                               shift_ranges=shift_ranges,
                               zoom_ranges=zoom_ranges,
                               rotation_ranges=rotation_ranges,
                               shear_ranges=shear_ranges,
                               noise_threshold=noise_threshold,
                               intensity_threshold=intensity_threshold,
                               blur_threshold=blur_threshold,
                               blur_probability=blur_probability,
                               crop_box_sides=crop_box_sides)

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
                              'describing the fraction of shearing allowed '
                              'both in the positive and negativ direction '
                              'of the given axis'))
    parser.add_argument('-n', '--noise_threshold', required=False,
                        default=None, type=float,
                        help=('Noise threshold used by the augmenter. If '
                             'used, the entire image is multiplied by a '
                             'uniform distribution from '
                             '[1-threshold, 1+threshold]'))
    parser.add_argument('-i', '--intensity_threshold', required=False,
                        default=None, type=float,
                        help=('Intensity threshold used by the augmenter. If '
                              'used, changes the intensity of the image by a '
                              'factor of [1+threshold, 1-threshold]'))
    parser.add_argument('-b', '--blur_threshold', required=False,
                        default=None, type=float,
                        help=('Blur sigma threshold used by the augmenter. If '
                              'used, blurs the image by a sigma in the range '
                              '[0, blur_threshold] with a probability given '
                              'by blur_probability'))
    parser.add_argument('-bp', '--blur_probability', required=False,
                        default=None, type=float,
                        help=('Sets the blur probability of the augmenter. If '
                              'used, defines the probability that a blur, '
                              'according to the blur_threshold, is applied. '
                              'If used without blur_threshold, has no effect'))
    parser.add_argument('-c', '--crop_box_sides', required=False, default=None,
                        type=int,
                        help=('Maxmimum sides of the lengths of the boxes '
                              'which are cropped by the augmenter. If set, '
                              'the augmenter will randomly crop out a '
                              'cuboid with sides [0-c, 0-c, 0-c] from each '
                              'image and fill it with noise'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where augmenter is stored')

    args = parser.parse_args()

    configure_nifti_augmenter(flip_probabilities=args.flip_probabilities,
                              shift_ranges=args.shift_ranges,
                              zoom_ranges=args.zoom_ranges,
                              rotation_ranges=args.rotation_ranges,
                              shear_ranges=args.shear_ranges,
                              noise_threshold=args.noise_threshold,
                              intensity_threshold=args.intensity_threshold,
                              blur_threshold=args.blur_threshold,
                              blur_probability=args.blur_probability,
                              crop_box_sides=args.crop_box_sides,
                              destination=args.destination)

"""Configures a NiftiPreprocessor

Example run:
    python scripts/configure_nifti_preprocessor.py \
        --sigma 255. \
        --destination /path/to/destination
"""

import argparse

from utils import configure_environment

configure_environment()

from pyment.data.preprocessors import NiftiPreprocessor


def configure_nifti_preprocessor(*, sigma: float = None, destination: str):
    preprocessor = NiftiPreprocessor(sigma)

    preprocessor.save(destination)

    return preprocessor


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Configures a NiftiPreprocessor')

    parser.add_argument('-s', '--sigma', required=False, default=None,
                        type=float,
                        help='Normalization constant used by the preprocessor')
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where preprocessor is stored')

    args = parser.parse_args()

    configure_nifti_preprocessor(sigma=args.sigma,
                                 destination=args.destination)

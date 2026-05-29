import argparse
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialImage
from tqdm import tqdm

from pyment.loaders.mgh import load_mgh

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def locate_images(directory: str, regex: str) -> list[str]:
    paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.match(regex, file):
                paths.append(os.path.join(root, file))

    return paths


def verify_image(path: str):
    logger.debug('Sanity checking image %s', path)

    tf_image = load_mgh(path)
    nib_image = nib.load(path)
    nib_image = cast(SpatialImage, nib_image)

    if not np.array_equal(tf_image, nib_image.get_fdata()):
        logger.error('TensorFlow and nibabel images do not match for %s', path)
        return False

    return True


def verify_mgh_loader(directory: str, regex: str, size: int, num_threads: int):
    paths = locate_images(directory, regex)
    logger.info('Found %d images to sanity check', len(paths))

    if size is not None:
        paths = paths[:size]
        logger.info('Restricted sanity check to first %d images', len(paths))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list(
            tqdm(executor.map(verify_image, paths), total=len(paths))
        )

    failures = len(futures) - sum(futures)

    logger.info(
        '%d/%d images failed the sanity check',
        failures,
        len(futures),
    )

    if failures > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        'Sanity checks the tensorflow MGH loader by loading all images in a '
        'given directory and comparing the result with what is produced by '
        'nibabel.'
    )

    parser.add_argument(
        'directory',
        help=(
            'Path to directory containing mgh/mgz images to load. The '
            'directory will be recursively searched for images matching the '
            'given regex, and all matching images will be loaded.'
        ),
    )
    parser.add_argument(
        '-r',
        '--regex',
        required=False,
        default=r'.*\.(?:mgh|mgz)$',
    )
    parser.add_argument(
        '-n',
        '--size',
        required=False,
        default=None,
        help=(
            'Number of images to load. If not specified, all images will be '
            'loaded.'
        ),
    )
    parser.add_argument(
        '-nt',
        '--num-threads',
        type=int,
        required=False,
        default=os.cpu_count(),
        help=(
            'Number of threads to use for the sanity check. If not specified, '
            'the number of available CPU cores will be used.'
        ),
    )

    args = parser.parse_args()

    verify_mgh_loader(
        args.directory,
        regex=args.regex,
        size=args.size,
        num_threads=args.num_threads,
    )


if __name__ == '__main__':
    main()

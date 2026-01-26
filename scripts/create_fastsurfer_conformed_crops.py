import argparse
import logging
import os

from pyment.data.utils.ensure_fastsurfer_crops_exists import (
    ensure_fastsurfer_crops_exists
)


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_fastsurfer_conformed_crops(
    root: str,
    target_shape: tuple[int, int, int],
    num_threads: int
) -> None:
    folders = os.listdir(root)
    logger.info('Found %d folders in %s', len(folders), root)
    folders = [
        os.path.join(root, folder) for folder in folders
        if os.path.isdir(os.path.join(root, folder, 'mri'))
    ]
    logger.info(
        'Found %d valid folders with mri subfolder in %s', len(folders), root
    )

    ensure_fastsurfer_crops_exists(
        folders,
        target_shape=target_shape,
        num_threads=num_threads
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Creates conformed crops for preprocessed FastSurfer images. The '
        'script will parse a FastSurfer output directory (i.e. one folder per '
        'processed image, with an \'mri\' subfolder), and ensure there exists '
        'a \'crop.mgz\' file in each subfolder. If the file does not exist, '
        'it will be created by combining the original image (\'orig.mgz\') '
        'and brain mask (\'mask.mgz\') and conforming the result.'
    )

    parser.add_argument(
        'root',
        help='Path to FastSurfer output directory'
    )
    parser.add_argument(
        '-ts', '--target_shape',
        nargs=3,
        type=int,
        required=False,
        default=(224, 192, 224),
        help='Target shape of the crop'
    )
    parser.add_argument(
        '-nt', '--num_threads',
        type=int,
        default=os.cpu_count(),
        help=(
            'Number of threads to use for the sanity check. If not specified, '
            'the number of available CPU cores will be used.'
        )
    )

    args = parser.parse_args()

    create_fastsurfer_conformed_crops(
        args.root,
        target_shape=args.target_shape,
        num_threads=args.num_threads
    )

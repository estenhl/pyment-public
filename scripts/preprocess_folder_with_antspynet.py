import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from preprocess_with_antspynet import preprocess_with_antspynet
from tqdm import tqdm

from pyment.utils.strip_extension import strip_extension

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def preprocess_folder_with_antspynet(
    input: str, output: str, suffix: str = '.mgz', num_threads: int = 1
) -> None:
    """Preprocesses every image in input with antspynet, in parallel.

    Parameters
    ----------
    input : str
        Path to a folder of images to preprocess.
    output : str
        Path to a folder where preprocessed images are written.
        Created if it does not already exist.
    suffix : str
        Extension (including leading dot) used for output filenames.
    num_threads : int
        Maximum number of threads to use for parallel processing.
    """
    os.makedirs(output, exist_ok=True)

    filenames = [
        filename
        for filename in os.listdir(input)
        if os.path.isfile(os.path.join(input, filename))
    ]
    logger.info('Found %d images in %s', len(filenames), input)

    def _process(filename: str) -> None:
        source = os.path.join(input, filename)
        destination = os.path.join(
            output, f'{strip_extension(filename)}{suffix}'
        )
        preprocess_with_antspynet(source, destination)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(_process, filenames), total=len(filenames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Preprocesses every image in a folder by performing '
        'skullstripping with antspynet, resampling to 1mm x 1mm x 1mm '
        'isotropic voxel sizes, reorienting to RSP space, and cropping '
        'a cube containing minimal non-brain tissue'
    )

    parser.add_argument('input', help='Path to folder of images to preprocess')
    parser.add_argument(
        'output', help='Path to folder where preprocessed images are written'
    )
    parser.add_argument(
        '-s',
        '--suffix',
        default='.mgz',
        help="Extension used for output filenames. Defaults to '.mgz'",
    )
    parser.add_argument(
        '-nt',
        '--num_threads',
        type=int,
        default=os.cpu_count(),
        help=(
            'Number of threads to use for preprocessing. If not specified, '
            'the number of available CPU cores will be used.'
        ),
    )

    args = parser.parse_args()

    preprocess_folder_with_antspynet(
        args.input,
        args.output,
        suffix=args.suffix,
        num_threads=args.num_threads,
    )

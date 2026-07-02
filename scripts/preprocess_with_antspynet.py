import argparse
import logging

import ants

from pyment.preprocessing import skullstrip_with_antspynet

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def preprocess_with_antspynet(input: str, output: str) -> None:
    """Skullstrips, resamples and reorients a nifti image with
    antspynet.

    Parameters
    ----------
    input : str
        Path to the image to preprocess.
    output : str
        Path where the preprocessed image is written.
    """
    image = ants.image_read(input)

    brain = skullstrip_with_antspynet(image)
    logger.debug('Preprocessed image orientation: %s', brain.orientation)

    ants.image_write(brain, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Preprocesses a nifti image by performing skullstripping with '
        'antspynet, resampling to 1mm x 1mm x 1mm isotropic voxel sizes, '
        'reorienting to RSP space, and cropping a cube containing minimal '
        'non-brain tissue'
    )

    parser.add_argument(
        '-i', '--input', required=True, help='Path to image to preprocess'
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Path where preprocessed image is written',
    )

    args = parser.parse_args()

    preprocess_with_antspynet(args.input, args.output)

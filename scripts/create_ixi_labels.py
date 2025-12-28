import argparse
import logging
import os
import re
import numpy as np
import pandas as pd


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def _parse_filename(filename: str) -> int:
    match = re.fullmatch(r'IXI(?P<id>\d+)-.*-T1\.nii\.gz', filename)

    if not match:
        raise ValueError(f'Unable to parse filename {filename}')

    return int(match.group('id'))

def _parse_images(folder: str) -> dict[int, str]:
    return {
        _parse_filename(filename): filename.split('.')[0]
        for filename in os.listdir(folder)
    }

def create_ixi_labels(source: str, destination: str, images: str):
    labels = pd.read_excel(source)

    images = _parse_images(images)
    logger.info(f'Read {len(labels)} labels')

    labels['image_id'] = labels['IXI_ID'].map(images)
    labels = labels[~labels['image_id'].isna()]
    logger.info(f'Found {len(labels)} labels with valid image paths')

    # Creates a binary bachelor label by encoding single as 1, and married,
    # divorced, cohabiting, and widowed as 0
    labels['bachelor'] = labels['MARITAL_ID'].map({
        0: np.nan, 1: True, 2: False, 3: False, 4: False, 5: False
    })

    # Directly copies height as a continuous label
    labels['height'] = labels['HEIGHT']

    # Encodes the original education label as a multiclass label
    labels['education'] = labels['QUALIFICATION_ID'].map({
        1: 'None',
        2: 'O-levels',
        3: 'A-levels',
        4: 'Further education',
        5: 'University'
    })

    columns = ['image_id', 'bachelor', 'height', 'education']
    labels[columns].to_csv(destination, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Compiles labels for the IXI dataset that can be used for finetuning. '
        'The labels in the dataset are more or less arbitrary encodings of '
        'original IXI demographics, constructed to demonstrate different '
        'finetuning approaches. Should not be used for actual research.'
    )

    parser.add_argument(
        '-s', '--source',
        required=True,
        help='Path to original IXI.xls file as downloaded from the IXI website'
    )
    parser.add_argument(
        '-d', '--destination',
        required=True,
        help='Path where new labels are written'
    )
    parser.add_argument(
        '-i', '--images',
        required=True,
        help='Path to folder containing IXI images'
    )

    args = parser.parse_args()

    create_ixi_labels(
        source=args.source,
        destination=args.destination,
        images=args.images
    )

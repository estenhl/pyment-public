import argparse
import math
import os
import requests
import tarfile
from tqdm import tqdm

from pyment.utils.download_file import download_file


DEFAULT_DESTINATION = os.path.join(os.path.expanduser('~'), 'data', 'ixi')

def download_tar(tar_path: str) -> str:
    url = (
        'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/'
        'IXI-T1.tar'
    )
    download_file(url, tar_path, 'Downloading T1 images')

def download_metadata(destination: str) -> str:
    url = (
        'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls'
    )
    download_file(url, destination, 'Downloading metadata')

def extract_images(tar_path: str, destination: str):
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(destination)

def download_ixi(destination: str):
    if os.path.isdir(destination):
        raise ValueError(f'Destination folder {destination} already exists')

    os.mkdir(destination)
    images_folder = os.path.join(destination, 'images')
    metadata_path = os.path.join(destination, 'IXI.xls')
    tar_path = os.path.join(destination, 'T1.tar.gz')

    download_tar(tar_path)
    download_metadata(metadata_path)
    extract_images(tar_path, images_folder)

    os.remove(tar_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Downloads T1 images and demographic data fromthe IXI dataset'
    )
    parser.add_argument(
        '-d', '--destination',
        required=False,
        default=DEFAULT_DESTINATION,
        help=(
            'Folder where the data will be downloaded. If this folder does '
            'not exist, the script will throw an error'
        )
    )

    args = parser.parse_args()

    download_ixi(args.destination)

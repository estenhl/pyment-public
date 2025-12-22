import argparse
import math
import os
import requests
import tarfile
from tqdm import tqdm


DEFAULT_DESTINATION = os.path.join(os.path.expanduser('~'), 'data', 'ixi')

def download_file(
    url: str,
    destination: str,
    description: str = None,
    decode_github: bool = False
) -> str:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        # 1 MB chunks
        chunk_size = 1<<20

        progress_bar = tqdm(
            response.iter_content(chunk_size=chunk_size),
            total=int(math.ceil(total_size / chunk_size)),
            unit='mb',
            unit_scale=True,
            unit_divisor=1024,
            desc=description
        )
        progress_bar.format_dict['rate'] = f'mb/s'

        with open(destination, 'wb') as f:
            for chunk in progress_bar:
                f.write(chunk)

    if decode_github:
        # Assumes a JSON file downloaded from GitHub
        with open(destination, 'rb') as f:
            data = json.load(f)

        data = base64.b64decode(data['content'])

        with open(destination, 'wb') as f:
            f.write(data)


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
        'Downloads T1 images and demographic data from the IXI dataset'
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

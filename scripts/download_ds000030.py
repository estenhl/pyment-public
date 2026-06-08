import argparse
import io
import math
import os

import pandas as pd
import requests
from tqdm import tqdm

DATASET_URL = 'https://s3.amazonaws.com/openneuro.org/ds000030'
PARTICIPANTS_URL = f'{DATASET_URL}/participants.tsv'
DEFAULT_DESTINATION = os.path.join(os.path.expanduser('~'), 'data', 'ds000030')


def _download_file(url: str, destination: str, description: str | None = None):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1 << 20

        progress_bar = tqdm(
            response.iter_content(chunk_size=chunk_size),
            total=int(math.ceil(total_size / chunk_size)),
            unit='mb',
            unit_scale=True,
            unit_divisor=1024,
            desc=description,
        )

        with open(destination, 'wb') as f:
            for chunk in progress_bar:
                f.write(chunk)


def _fetch_participants() -> pd.DataFrame:
    response = requests.get(PARTICIPANTS_URL)
    response.raise_for_status()
    participants = pd.read_csv(io.StringIO(response.text), sep='\t')
    return participants[participants['T1w'] == 1].reset_index(drop=True)


def download_dataset(destination: str, n: int | None = None):
    if os.path.isdir(destination):
        raise ValueError(f'Destination folder {destination} already exists')

    os.makedirs(destination)
    images_folder = os.path.join(destination, 'images')
    os.makedirs(images_folder)

    participants = _fetch_participants()

    if n is not None:
        participants = participants.iloc[:n]

    for _, row in participants.iterrows():
        subject = row['participant_id']
        url = f'{DATASET_URL}/{subject}/anat/{subject}_T1w.nii.gz'
        dest = os.path.join(images_folder, f'{subject}_T1w.nii.gz')
        _download_file(url, dest, description=subject)

    labels = participants[
        ['participant_id', 'age', 'gender', 'diagnosis']
    ].copy()
    labels['id'] = labels['participant_id'] + '_T1w'
    labels = labels.drop(columns='participant_id')
    labels = labels.rename(columns={'gender': 'sex'})
    labels['has_diagnosis'] = labels['diagnosis'] != 'CONTROL'
    labels = labels[['id', 'age', 'sex', 'diagnosis', 'has_diagnosis']]
    labels.to_csv(os.path.join(destination, 'labels.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Downloads T1 images and demographic labels from ds000030 (OpenNeuro)'
    )
    parser.add_argument(
        '-d',
        '--destination',
        required=False,
        default=DEFAULT_DESTINATION,
        help=(
            'Folder where the data will be downloaded (must not already exist)'
        ),
    )
    parser.add_argument(
        '-n',
        '--num-subjects',
        required=False,
        default=None,
        type=int,
        help='Number of subjects to download (default: all 265 with T1w)',
    )

    args = parser.parse_args()

    download_dataset(args.destination, n=args.num_subjects)

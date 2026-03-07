import argparse
import logging
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple

import nibabel as nib

from pyment.data.utils import ensure_fastsurfer_crop_exists
from pyment.models.sfcn import sfcn_factory
from pyment.preprocessing.conform import conform


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def _parse_folder_name(name: str) -> Tuple[str, str, str]:
    match = re.fullmatch(r'sub-(.*)_ses-(.*)_run-([^_]).*', name)

    if not match:
        return None, None, None

    return match.groups()

def predict_from_fastsurfer_folder(
    source: str,
    folders: List[str] = None,
    weights: str = None,
    model_name: str = 'sfcn-multi',
    targets: List[str] = [
        'age', 'sex', 'handedness', 'bmi', 'fluid_intelligence', 'neuroticism'
    ],
    destination: str = None
) -> pd.DataFrame:
    if destination is not None and os.path.isfile(destination):
        raise ValueError(f'Destination {destination} already exists')

    logger.info('Loading multi-task model with weights %s', weights)

    model_class = sfcn_factory(model_name)
    model = model_class(weights=weights)

    results = []

    logger.info(f'Reading fastsurfer folders from {source}')

    folders = (
        folders if folders is not None
        else [
            folder for folder in os.listdir(source)
            if os.path.isdir(os.path.join(source, folder))
        ]
    )

    for folder in tqdm(folders):
        subject, session, run = _parse_folder_name(folder)

        if not ensure_fastsurfer_crop_exists(
            os.path.join(source, folder),
            target_shape=(224, 192, 224)
        ):
            logger.warning(
                'Unable to generate prediction for %s: Can\'t ensure crop '
                'exists',
                folder
            )
            continue

        image = os.path.join(source, folder, 'mri', 'crop.mgz')
        image = nib.load(image).get_fdata()

        predictions = model.predict(
            np.expand_dims(image, axis=0),
            verbose=0
        )[0]
        logger.debug('Predictions for %s: %s', folder, str(predictions))

        results.append({
            **{
                'source': os.path.join(source, folder),
                'subject': subject,
                'session': session,
                'run': run
            },
            **{targets[i]: predictions[i] for i in range(len(targets))}
        })

    results = pd.DataFrame(results)

    if destination is not None:
        results.to_csv(destination, index=False)

    return results

def main():
    parser = argparse.ArgumentParser(
        'Generates multi-task predictions for preprocessed images organized '
        'in a FastSurfer folder'
    )

    parser.add_argument(
        'root',
        help=(
            'Path to FastSurfer folder. Should contain subfolders that have '
            'an \'mri\' subfolder that contains files orig.mgz and mask.mgz'
        )
    )
    parser.add_argument(
        '-w', '--weights',
        required=False,
        default='multi-2025',
        help=(
            'Weights to use. Should either point to a local file path, or a '
            'known identifier. If a local file path <path> is used, there '
            'should exist files named <path>.index and '
            '<path>.data-00000-of-00001. Defaults to \'multi-2025\''
        )
    )
    parser.add_argument(
        '-m', '--model',
        required=False,
        default='sfcn-multi',
        help=(
            'Name of the model to use. Defaults to sfcn-multi'
        )
    )
    parser.add_argument(
        '-t', '--targets',
        required=False,
        nargs='+',
        default=[
            'age', 'sex', 'handedness', 'bmi', 'fluid_intelligence',
            'neuroticism'
        ],
        help=(
            'Name to use for each of the prediction heads in the output CSV. '
            'Defaults to the target labels for the \'multi-2025\' model'
        )
    )
    parser.add_argument(
        '-f', '--folders',
        default=None,
        nargs='+',
        help=(
            'List of folders to process. If not provided, all folders in '
            'the source folder will be processed.'
        )
    )
    parser.add_argument(
        '-d', '--destination',
        required=False,
        default=None,
        help='Path where CSV with predictions are written'
    )

    args = parser.parse_args()

    predict_from_fastsurfer_folder(
        source=args.root,
        folders=args.folders,
        model_name=args.model,
        weights=args.weights,
        targets=args.targets,
        destination=args.destination,
    )

if __name__ == '__main__':
    main()


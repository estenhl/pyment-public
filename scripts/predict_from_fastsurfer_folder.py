import argparse
import logging
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple

import nibabel as nib

from pyment.models import MultiTaskSFCN
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
    weights: str, 
    destination: str = None
) -> pd.DataFrame:
    if destination is not None and os.path.isfile(destination):
        raise ValueError(f'Destination {destination} already exists')
    
    logger.info('Loading multi-task model with weights %s', weights)
    model = MultiTaskSFCN(weights=weights)

    results = []

    for folder in tqdm(os.listdir(source)):
        orig = os.path.join(source, folder, 'mri', 'orig.mgz')

        subject, session, run = _parse_folder_name(folder)

        if not os.path.isfile(orig):
            logger.warning('No orig.mgz file for folder %s', folder)
            continue

        orig = nib.load(orig)
        orig = orig.get_fdata()
        brainmask = os.path.join(source, folder, 'mri', 'mask.mgz')

        if not os.path.isfile(brainmask):
            logger.warning('No mask.mgz file for folder %s', folder)
            continue
        
        brainmask = nib.load(brainmask)
        brainmask = brainmask.get_fdata()

        image = orig * brainmask

        logger.debug('Conforming image from %s', os.path.join(source, folder))
        image = conform(image)

        predictions = model.predict(np.expand_dims(image, axis=0))[0]
        logger.debug('Predictions for %s: %s', folder, str(predictions))
        
        results.append({
            'source': os.path.join(source, folder),
            'subject': subject,
            'session': session,
            'run': run,
            'age': predictions[0],
            'sex': predictions[1],
            'handedness': predictions[2],
            'bmi': predictions[3],
            'fluid_intelligence': predictions[4],
            'neuroticism': predictions[5]
        })

    results = pd.DataFrame(results)

    if destination is not None:
        results.to_csv(destination, index=False)

    return results

if __name__ == '__main__':
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
        required=True,
        help=(
            'Weights to use. Should either point to a local file path, or a '
            'known keyword. If a local file path <path> is used, there should '
            'exist files named <path>.index and <path>.data-00000-of-00001'
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
        weights=args.weights,
        destination=args.destination
    )


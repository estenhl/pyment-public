import argparse
import logging
import os
import re
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

import nibabel as nib

from pyment.models.sfcn import sfcn_factory
from pyment.preprocessing.conform import conform


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def _extract_run(filename: str) -> str:
    match = re.fullmatch(
        r'.*_run-([^_]+)(?:_.*)?(?:\.nii(?:\.gz)?|\.mgz)', 
        filename
    )

    if not match:
        logger.warning('Unable to extract run for filename %s', filename)
        return None

    return match.groups()[0]

def _extract_modality(filename: str) -> str:
    match = re.fullmatch(
        r'.*_run-(?:[^_]+)(?:_(.*))?(?:\.nii(?:\.gz)?|\.mgz)', 
        filename
    )

    if not match:
        logger.warning('Unable to extract modality for filename %s', filename)
        return None

    return match.groups()[0]
def predict_from_bids_folder(
    source: str, 
    weights: str,
    model_name: str = 'sfcn-multi',
    targets: List[str] = [
        'age', 'sex', 'handedness', 'bmi', 'fluid_intelligence', 'neuroticism'
    ],
    destination: str = None,
    per_image_normalization: bool = False
) -> pd.DataFrame:
    if destination is not None and os.path.isfile(destination):
        raise ValueError(f'Destination {destination} already exists')
    
    logger.info('Loading multi-task model with weights %s', weights)
    model_class = sfcn_factory(model_name)
    model = model_class(weights=weights)

    results = []

    for subject in tqdm(os.listdir(source)):
        for session in os.listdir(os.path.join(source, subject)):
            anat_folder = os.path.join(source, subject, session, 'anat')

            if not os.path.isdir(anat_folder):
                logger.warning(
                    'No anat-folder exists for subject %s and session %s',
                    subject, session
                )
                continue

            for filename in os.listdir(anat_folder):
                path = os.path.join(anat_folder, filename)
                run = _extract_run(filename)
                modality = _extract_modality(filename)

                logger.debug(f'Loading image {path}')
                image = nib.load(os.path.join(anat_folder, filename))
                image = image.get_fdata()

                logger.debug(f'Conforming image {path}')
                image = conform(
                    image, 
                    relative_normalization=per_image_normalization
                )

                predictions = model.predict(np.expand_dims(image, axis=0))[0]
                logger.debug(
                    'Predictions for %s, %s: %s', 
                    subject, session, str(predictions)
                )

                results.append({
                    **{
                        'source': path,
                        'subject': subject.replace('sub-', ''),
                        'session': session.replace('ses-', ''),
                        'run': run,
                        'modality': modality
                    },
                    **{targets[i]: predictions[i] for i in range(len(targets))}
                })

    results = pd.DataFrame(results)

    if destination is not None:
        results.to_csv(destination, index=False)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Generates multi-task predictions for preprocessed images organized '
        'in a BIDS folder'
    )

    parser.add_argument('bids', help='Path to BIDS folder')
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
        '-m', '--model', 
        required=False,
        default='sfcn-multi',
        help=(
            'Name of the model to use'
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
        help='Name to use for each of the prediction heads in the output CSV'
    )
    parser.add_argument(
        '-d', '--destination',
        required=False,
        default=None,
        help='Path where CSV with predictions are written'
    )
    parser.add_argument(
        '-n', '--per_image_normalization',
        action='store_true',
        help=(
            'If set, the voxel values of each individual image is conformed '
            'by dividing by the image max instead of the constant 255'
        )
    )

    args = parser.parse_args()

    predict_from_bids_folder(
        source=args.bids,
        weights=args.weights,
        model_name=args.model,
        targets=args.targets,
        destination=args.destination,
        per_image_normalization=args.per_image_normalization
    )
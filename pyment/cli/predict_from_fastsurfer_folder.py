"""Generate predictions for FastSurfer subject folders."""

import argparse
import logging
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.spatialimages import SpatialImage
from tqdm import tqdm

from pyment.data.utils import ensure_fastsurfer_crop_exists
from pyment.models.sfcn import sfcn_factory

logger = logging.getLogger(__name__)


def _parse_folder_name(name: str) -> tuple[str, str, str]:
    match = re.fullmatch(
        r'sub-(?P<subject>.*)_ses-(?P<session>.*)_run-(?P<run>[^_]).*', name
    )

    if not match:
        raise ValueError(f'Unable to match {name}')

    return (match.group('subject'), match.group('session'), match.group('run'))


def predict_from_fastsurfer_folder(
    source: str,
    folders: list[str] | None = None,
    weights: str | None = None,
    model_name: str = 'sfcn-multi',
    targets: list[str] = [
        'age',
        'sex',
        'handedness',
        'bmi',
        'fluid_intelligence',
        'neuroticism',
    ],
    destination: str | None = None,
) -> pd.DataFrame:
    """Generate predictions for FastSurfer subject folders.

    Iterates folders under ``source``
    (or the explicit ``folders`` list), lazily generates the model crop
    if missing, runs the model, and returns one row per processed
    folder. Folders whose names don't match ``sub-X_ses-Y_run-Z``
    produce ``None`` for the ``subject``, ``session``, and ``run``
    metadata columns.

    Parameters
    ----------
    source : str
        Path to the FastSurfer output root containing subject folders.
    folders : list[str] | None, optional
        Specific subfolder names under ``source`` to process. If
        ``None``, all subdirectories of ``source`` are processed.
    weights : str | None, optional
        Path prefix to local weight files or a known identifier (e.g.
        ``'multi-2025'``).
    model_name : str, optional
        Name dispatched via ``sfcn_factory``. Defaults to
        ``'sfcn-multi'``.
    targets : list[str], optional
        Column names for each prediction head in the output. Defaults to
        the six heads of the ``multi-2025`` model in order.
    destination : str | None, optional
        Path to write predictions as CSV. If ``None``, no file is
        written.

    Returns
    -------
    pd.DataFrame
        One row per processed folder, with metadata columns
        (``source``, ``subject``, ``session``, ``run``) and one column
        per entry in ``targets``.

    Raises
    ------
    ValueError
        If ``destination`` already exists as a file.
    """

    if destination is not None and os.path.isfile(destination):
        raise ValueError(f'Destination {destination} already exists')

    logger.info('Loading multi-task model with weights %s', weights)

    model_class = sfcn_factory(model_name)
    model = model_class(weights=weights)

    results = []

    logger.info('Reading fastsurfer folders from %s', source)

    folders = (
        folders
        if folders is not None
        else [
            folder
            for folder in os.listdir(source)
            if os.path.isdir(os.path.join(source, folder))
        ]
    )

    for folder in tqdm(folders):
        try:
            subject, session, run = _parse_folder_name(folder)
        except ValueError as e:
            logger.warning(str(e))
            subject, session, run = None, None, None

        if not ensure_fastsurfer_crop_exists(os.path.join(source, folder)):
            logger.warning(
                "Unable to generate prediction for %s: Can't ensure crop "
                'exists',
                folder,
            )
            continue

        image_path = os.path.join(source, folder, 'mri', 'crop.mgz')
        loaded = nib.load(image_path)
        assert isinstance(loaded, SpatialImage)
        image_data = loaded.get_fdata()

        predictions = model.predict(
            np.expand_dims(image_data, axis=0), verbose=0
        )[0]
        logger.debug('Predictions for %s: %s', folder, str(predictions))

        results.append(
            {
                **{
                    'source': os.path.join(source, folder),
                    'subject': subject,
                    'session': session,
                    'run': run,
                },
                **{targets[i]: predictions[i] for i in range(len(targets))},
            }
        )

    results = pd.DataFrame(results)

    if destination is not None:
        results.to_csv(destination, index=False)

    return results


def main() -> None:
    """Entry point for the ``pyment-predict`` CLI."""

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        'Generates multi-task predictions for preprocessed images organized '
        'in a FastSurfer folder'
    )

    parser.add_argument(
        'root',
        help=(
            'Path to FastSurfer folder. Should contain subfolders that have '
            "an 'mri' subfolder that contains files orig.mgz and mask.mgz"
        ),
    )
    parser.add_argument(
        '-w',
        '--weights',
        required=False,
        default='multi-2025',
        help=(
            'Weights to use. Should either point to a local file path, or a '
            'known identifier. If a local file path <path> is used, there '
            'should exist files named <path>.index and '
            "<path>.data-00000-of-00001. Defaults to 'multi-2025'"
        ),
    )
    parser.add_argument(
        '-m',
        '--model',
        required=False,
        default='sfcn-multi',
        help=('Name of the model to use. Defaults to sfcn-multi'),
    )
    parser.add_argument(
        '-t',
        '--targets',
        required=False,
        nargs='+',
        default=[
            'age',
            'sex',
            'handedness',
            'bmi',
            'fluid_intelligence',
            'neuroticism',
        ],
        help=(
            'Name to use for each of the prediction heads in the output CSV. '
            "Defaults to the target labels for the 'multi-2025' model"
        ),
    )
    parser.add_argument(
        '-f',
        '--folders',
        default=None,
        nargs='+',
        help=(
            'List of folders to process. If not provided, all folders in '
            'the source folder will be processed.'
        ),
    )
    parser.add_argument(
        '-d',
        '--destination',
        required=False,
        default=None,
        help='Path where CSV with predictions are written',
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

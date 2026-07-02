"""Generate predictions for preprocessed image folders."""

from __future__ import annotations

import argparse
import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TARGETS = [
    'age',
    'sex',
    'handedness',
    'bmi',
    'fluid_intelligence',
    'neuroticism',
]


def _parse_folder_name(
    name: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    match = re.fullmatch(
        r'sub-(?P<subject>[^_]+)'
        r'(?:_ses-(?P<session>[^_]+))?'
        r'(?:_run-(?P<run>[^_]+))?'
        r'(?:_(?P<modality>[^_-]+))?',
        name,
    )

    if not match:
        raise ValueError(f'Unable to match {name}')

    return (
        match.group('subject'),
        match.group('session'),
        match.group('run'),
        match.group('modality'),
    )


def _detect_format(source: str) -> str:
    """Infers whether source is a FastSurfer or a flat image folder.

    A FastSurfer folder holds one subdirectory per subject, each
    containing an 'mri' subdirectory. A flat folder holds one image
    file per subject directly.

    Parameters
    ----------
    source : str
        Path to the folder to inspect.

    Returns
    -------
    str
        Either 'fastsurfer' or 'flat'.

    Raises
    ------
    ValueError
        If source is empty, mixes files and subdirectories, or holds
        subdirectories that don't all contain an 'mri' subdirectory.
    """
    entries = os.listdir(source)

    if not entries:
        raise ValueError(f'Unable to infer folder format for {source}: empty')

    paths = [os.path.join(source, entry) for entry in entries]

    if all(os.path.isfile(path) for path in paths):
        return 'flat'

    if all(os.path.isdir(path) for path in paths):
        if all(os.path.isdir(os.path.join(path, 'mri')) for path in paths):
            return 'fastsurfer'

        raise ValueError(
            f'Unable to infer folder format for {source}: subdirectories '
            "exist but not all contain an 'mri' subdirectory"
        )

    raise ValueError(
        f'Unable to infer folder format for {source}: contains a mix of '
        'files and subdirectories'
    )


def predict_from_fastsurfer_folder(
    source: str,
    folders: list[str] | None = None,
    weights: str | None = None,
    model_name: str = 'sfcn-multi',
    targets: list[str] = _DEFAULT_TARGETS,
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

    import nibabel as nib
    import numpy as np
    from nibabel.spatialimages import SpatialImage
    from tqdm import tqdm

    from pyment.data.utils import ensure_fastsurfer_crop_exists
    from pyment.models.sfcn import sfcn_factory

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
            subject, session, run, modality = _parse_folder_name(folder)
        except ValueError as e:
            logger.warning(str(e))
            subject, session, run, modality = None, None, None, None

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
                    'modality': modality,
                    'run': run,
                },
                **{targets[i]: predictions[i] for i in range(len(targets))},
            }
        )

    results = pd.DataFrame(results)

    if destination is not None:
        results.to_csv(destination, index=False)

    return results


def predict_from_flat_folder(
    source: str,
    files: list[str] | None = None,
    weights: str | None = None,
    model_name: str = 'sfcn-multi',
    targets: list[str] = _DEFAULT_TARGETS,
    target_shape: tuple[int, int, int] = (224, 192, 224),
    destination: str | None = None,
) -> pd.DataFrame:
    """Generate predictions for a flat folder of preprocessed images.

    Iterates files under ``source`` (or the explicit ``files`` list),
    zero-pads each image up to ``target_shape``, runs the model, and
    returns one row per processed file. Filenames whose stem (after
    stripping the extension) doesn't match ``sub-X_ses-Y_run-Z``
    produce ``None`` for the ``subject``, ``session``, and ``run``
    metadata columns.

    Parameters
    ----------
    source : str
        Path to a folder holding one preprocessed image file per
        subject.
    files : list[str] | None, optional
        Specific filenames under ``source`` to process. If ``None``,
        all files in ``source`` are processed.
    weights : str | None, optional
        Path prefix to local weight files or a known identifier (e.g.
        ``'multi-2025'``).
    model_name : str, optional
        Name dispatched via ``sfcn_factory``. Defaults to
        ``'sfcn-multi'``.
    targets : list[str], optional
        Column names for each prediction head in the output. Defaults to
        the six heads of the ``multi-2025`` model in order.
    target_shape : tuple[int, int, int], optional
        Shape each image is zero-padded up to before prediction.
    destination : str | None, optional
        Path to write predictions as CSV. If ``None``, no file is
        written.

    Returns
    -------
    pd.DataFrame
        One row per processed file, with metadata columns
        (``source``, ``subject``, ``session``, ``run``) and one column
        per entry in ``targets``.

    Raises
    ------
    ValueError
        If ``destination`` already exists as a file, or if a file
        exceeds ``target_shape`` along any axis.
    """

    import numpy as np
    from tqdm import tqdm

    from pyment.loaders.mgh import load_mgh
    from pyment.models.sfcn import sfcn_factory
    from pyment.utils.strip_extension import strip_extension

    if destination is not None and os.path.isfile(destination):
        raise ValueError(f'Destination {destination} already exists')

    logger.info('Loading multi-task model with weights %s', weights)

    model_class = sfcn_factory(model_name)
    model = model_class(weights=weights)

    results = []

    logger.info('Reading files from %s', source)

    files = (
        files
        if files is not None
        else [
            filename
            for filename in os.listdir(source)
            if os.path.isfile(os.path.join(source, filename))
        ]
    )

    for filename in tqdm(files):
        try:
            subject, session, run, modality = _parse_folder_name(
                strip_extension(filename)
            )
        except ValueError as e:
            logger.warning(str(e))
            subject, session, run, modality = None, None, None, None

        image_path = os.path.join(source, filename)
        image_data = load_mgh(image_path).numpy()

        if any(
            image_data.shape[axis] > target_shape[axis] for axis in range(3)
        ):
            raise ValueError(
                f'Image {filename} with shape {image_data.shape} exceeds '
                f'target_shape {target_shape}'
            )

        padding = [
            (0, target_shape[axis] - image_data.shape[axis])
            for axis in range(3)
        ]
        image_data = np.pad(image_data, padding)

        predictions = model.predict(
            np.expand_dims(image_data, axis=0), verbose=0
        )[0]
        logger.debug('Predictions for %s: %s', filename, str(predictions))

        results.append(
            {
                **{
                    'source': image_path,
                    'subject': subject,
                    'session': session,
                    'modality': modality,
                    'run': run,
                },
                **{targets[i]: predictions[i] for i in range(len(targets))},
            }
        )

    results = pd.DataFrame(results)

    if destination is not None:
        results.to_csv(destination, index=False)

    return results


def predict(
    source: str,
    format: str = 'auto',
    entries: list[str] | None = None,
    weights: str | None = None,
    model_name: str = 'sfcn-multi',
    targets: list[str] = _DEFAULT_TARGETS,
    destination: str | None = None,
) -> pd.DataFrame:
    """Generate predictions for a folder of preprocessed images.

    Dispatches to ``predict_from_fastsurfer_folder`` or
    ``predict_from_flat_folder`` depending on ``format``, inferring
    the folder structure of ``source`` via ``_detect_format`` when
    ``format`` is ``'auto'``.

    Parameters
    ----------
    source : str
        Path to a FastSurfer output root, or a flat folder of
        individually preprocessed images.
    format : str, optional
        One of ``'auto'``, ``'fastsurfer'``, ``'flat'``. Defaults to
        ``'auto'``.
    entries : list[str] | None, optional
        Specific subfolder names (FastSurfer format) or filenames
        (flat format) under ``source`` to process. If ``None``, all
        entries in ``source`` are processed.
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
        One row per processed subject.

    Raises
    ------
    ValueError
        If ``format`` is not one of ``'auto'``, ``'fastsurfer'`` or
        ``'flat'``.
    """

    if format not in ('auto', 'fastsurfer', 'flat'):
        raise ValueError(f'Unknown format {format}')

    resolved = _detect_format(source) if format == 'auto' else format
    logger.info('Using %s format for %s', resolved, source)

    if resolved == 'fastsurfer':
        return predict_from_fastsurfer_folder(
            source=source,
            folders=entries,
            weights=weights,
            model_name=model_name,
            targets=targets,
            destination=destination,
        )

    return predict_from_flat_folder(
        source=source,
        files=entries,
        weights=weights,
        model_name=model_name,
        targets=targets,
        destination=destination,
    )


def main() -> None:
    """Entry point for the ``pyment-predict`` CLI."""

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        'Generates multi-task predictions for preprocessed images, '
        'organized either in a FastSurfer folder or a flat folder of '
        'individually preprocessed images'
    )

    parser.add_argument(
        'root',
        help=(
            'Path to a folder of preprocessed images: either a FastSurfer '
            "output root (subfolders with an 'mri' subfolder containing "
            'orig.mgz and mask.mgz), or a flat folder with one image file '
            'per subject.'
        ),
    )
    parser.add_argument(
        '--format',
        choices=['auto', 'fastsurfer', 'flat'],
        default='auto',
        help=(
            "Folder format of root. Defaults to 'auto', which infers the "
            'format from the contents of root.'
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
        default=_DEFAULT_TARGETS,
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
            'List of subfolders (FastSurfer format) or filenames (flat '
            'format) to process. If not provided, all entries in root '
            'will be processed.'
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

    predict(
        source=args.root,
        format=args.format,
        entries=args.folders,
        model_name=args.model,
        weights=args.weights,
        targets=args.targets,
        destination=args.destination,
    )


if __name__ == '__main__':
    main()

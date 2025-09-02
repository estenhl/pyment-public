from __future__ import annotations

import logging
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Union

import nibabel as nib
from pydantic import model_validator, BaseModel

from .data_split_configuration import DataSplitConfiguration


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def _extract_run(filename: str) -> Union[str, None]:
    match = re.fullmatch(r'.*_run-(?P<run>[^_.]*)(?:_.*)?\.mgz', filename)

    if match:
        return match.group('run')

    logger.warning('Unable to extract run from filename %s', filename)

    return None

def _parse_bids_folder(root: str):
    entries = []

    for subject_folder in os.listdir(root):
        subject_match = re.fullmatch(r'sub-(?P<subject>.*)', subject_folder)
        
        if not subject_match:
            logger.warning(
                'Subject folder %s in %s does not have the expected sub-XXX '
                'format. Skipping', subject_folder, root
            )
            continue

        subject = subject_match.group('subject')

        for session_folder in os.listdir(os.path.join(root, subject_folder)):
            session_match = re.fullmatch(
                r'ses-(?P<session>.*)', session_folder
            )

            if not session_match:
                logger.warning(
                    'Session folder %s in subject %s in folder %s does not '
                    'match the expected ses-XXX format. Skipping', 
                    session_folder, subject_folder, root
                )
                continue

            session = session_match.group('session')

            anat_folder = os.path.join(
                root, subject_folder, session_folder, 'anat'
            )

            t1s = [
                filename for filename in os.listdir(anat_folder) 
                if 'T1' in filename
            ]

            for filename in t1s:
                run = _extract_run(filename)
                entries.append({
                    'subject': subject,
                    'session': session,
                    'run': run,
                    'path': os.path.join(anat_folder, filename)
                })

    return pd.DataFrame(entries, columns=['subject', 'session', 'run', 'path'])

def _parse_bids_folders(folders: List[str]):
    df = pd.concat([_parse_bids_folder(folder) for folder in folders])
    df = df.reset_index()
    logger.info('Parsed %d images', len(df))

    return df

def _parse_fastsurfer_name(name: str) -> Tuple[str, str, str]:
    match = re.fullmatch(r'sub-(.*)_ses-(.*)_run-(.*)(?:T1w?)?', name)

    if not match:
        raise ValueError(
            'Unable to extract subject, session, run from folder %s', name
        )

    return match.groups()

def _parse_fastsurfer_folder(folder: str):
    entries = []

    for subfolder in os.listdir(folder):
        subject, session, run = _parse_fastsurfer_name(subfolder)

        mri_folder = os.path.join(folder, subfolder, 'mri')
        brainmask = os.path.join(mri_folder, 'brainmask.mgz')

        if not os.path.isfile(brainmask):
            logger.info('Brainmask does not exist in folder %s', subfolder)

            orig = os.path.join(mri_folder, 'orig.mgz')
            mask = os.path.join(mri_folder, 'mask.mgz')

            if not os.path.isfile(orig):
                logger.error('Orig does not exist in folder %s', subfolder)
                continue
            elif not os.path.isfile(mask):
                logger.error('Mask does not exist in folder %s', subfolder)
                continue

            orig_data = nib.load(orig)
            mask_data = nib.load(mask)
            brainmask_data = nib.Nifti1Image(
                orig_data.get_fdata() * mask_data.get_fdata(), 
                header=orig_data.header, 
                affine=orig_data.affine
            )

            nib.save(brainmask_data, brainmask)
        
        entries.append({
            'subject': subject,
            'session': session,
            'run': run,
            'path': brainmask
        })

    return pd.DataFrame(entries, columns=['subject', 'session', 'run', 'path'])

def _parse_fastsurfer_folders(folders: List[str]):
    df = pd.concat([_parse_fastsurfer_folder(folder) for folder in folders])
    df = df.reset_index()
    logger.info('Parsed %d images', len(df))

    return df

def _summarize_values(values: np.ndarray, name: str):
    if not np.issubdtype(values.dtype, np.number):
        logger.info('%s: %s', name, Counter(values))
    elif np.array_equal(
        np.unique(values[~np.isnan(values)]), 
        np.asarray([0, 1])
    ):
        nans = len(np.where(np.isnan(values))[0])
        logger.info(
            '%s: %s (%d NAs)', name, Counter(values[~np.isnan(values)]), nans
        )
    else:
        nans = len(np.where(np.isnan(values))[0])
        mean = np.round(np.nanmean(values), 2)
        std = np.round(np.nanstd(values), 2)
        logger.info('%s: %.2f+/-%.2f (%d NAs)', name, mean, std, nans)

def _summarize(df: pd.DataFrame, variables: List[str], name: str):
    logger.info('%s n=%d', name, len(df))

    for variable in variables:
        _summarize_values(df[variable].values, name=variable)


def _split_training_validation_fold(
    df: pd.DataFrame, 
    labels: str, 
    training_fraction: float,
    target: str = None,
    stratification: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    columns = set(['subject', 'session', 'run'])
    
    if target:
        columns.add(target)
    
    if stratification:
        columns |= set(stratification)
    
    labels = pd.read_csv(
        labels,
        usecols=list(columns),
        dtype={'subject': object, 'session': object, 'run': object},
    )

    logger.info('Parsed %d labels', len(labels))

    if not len(labels) == len(labels.drop_duplicates(['subject', 'session'])):
        raise ValueError(
            f'There are duplicates (subject, session)-pairs in the labels file'
        )

    df = pd.merge(
       df, labels, 
       how='inner', 
       left_on=['subject', 'session'],
       right_on=['subject', 'session']
    )

    logger.info('Merged %d data points', len(df))

    if stratification is not None:
        df = df.sort_values(stratification)

    subjects = df.drop_duplicates('subject')
    num_folds = int(1.0 / (1 - training_fraction))

    if num_folds == 1:
        raise ValueError(
            'Training fraction %.2f yields a single fold', training_fraction
        )
    
    subjects['fold'] = np.arange(len(df)) % num_folds
    folds = {row['subject']: row['fold'] for _, row in subjects.iterrows()}
    df['fold'] = df['subject'].map(folds)

    validation_fold = num_folds // 2
    training = df[df['fold'] != validation_fold]
    validation = df[df['fold'] == validation_fold]

    if len(
        set(training['subject'].values) & set(validation['subject'].values)
    ) > 0:
        raise ValueError('Overlap between training and validation folds')

    if stratification:
        for name, df in [('Training', training), ('Validation', validation)]:
            _summarize(df, variables=stratification, name=name)
    
    return training, validation

class DatasetConfiguration(BaseModel):
    input_shape: Tuple[int, int, int]
    bids: List[str] | None = None
    fastsurfer: List[str] | None = None
    labels: str
    split: DataSplitConfiguration = None

    @model_validator(mode='after')
    def check_fastsurfer_or_bids(self):
        if self.bids is not None and self.fastsurfer is not None:
            raise ValueError(
                'Either \'bids\' or \'fastsurfer\'-property must be set, not '
                'both'
            )
        elif self.bids is None and self.fastsurfer is None:
            raise ValueError(
                'Either \'bids or \'fastsurfer\'-property must be set'
            )

        return self

    @staticmethod
    def parse(
        configuration: DatasetConfiguration, 
        target: str = None
    ) -> Dict[str, pd.DataFrame]:
        if configuration.split:
            if configuration.bids:
                df = _parse_bids_folders(configuration.bids)
            elif configuration.fastsurfer:
                df = _parse_fastsurfer_folders(configuration.fastsurfer)
            else:
                raise ValueError(
                    'Unable to parse DatasetConfiguration without either '
                    '\'bids\' or \'fastsurfer\' set'
                )
            
            return _split_training_validation_fold(
                df=df,
                labels=configuration.labels,
                training_fraction=configuration.split.training_fraction,
                target=target,
                stratification=configuration.split.stratification
            )

        raise NotImplementedError(
            f'Not sure how to parse dataset without a split configuration'
        )
from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd

from typing import Dict

from .dataset import Dataset


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

class NiftiDataset(Dataset):
    @classmethod
    def from_folder(cls, root: str, *, images: str = 'images', 
                    labels: str = 'labels.csv', suffix: str = 'nii.gz', 
                    **kwargs) -> NiftiDataset:
        images = os.path.join(root, images)
        labels = os.path.join(root, labels)

        df = pd.read_csv(labels, index_col=None)

        assert 'id' in df.columns, 'labels is missing id column'

        label_ids = set(df['id'])
        image_ids = set([filename.split('.')[0] \
                         for filename in os.listdir(images)])

        missing_images = label_ids - image_ids
        missing_labels = image_ids - label_ids
        complete = label_ids & image_ids

        for id in missing_images:
            logger.warning(f'Skipping {id}: Missing image')

        for id in missing_labels:
            logger.warning(f'Skipping {id}: Missing labels')

        df = df[df['id'].isin(complete)]

        if 'path' in df.columns:
            logger.warning((f'{labels} should not contain a field named '
                            '\'path\' as this is used internally. This column '
                            'will be dropped'))

        create_path = lambda x: os.path.join(images, f'{x}.{suffix}')

        df['path'] = df['id'].apply(create_path)

        paths = df['path'].values
        variables = [var for var in df.columns if var not in ['path', 'id']]
        labels = {var: df[var].values for var in variables}

        logger.debug((f'Creating {cls.__name__} with {len(df)} datapoints and '
                      f'labels {variables}'))

        return cls(paths, labels, **kwargs)

    @property
    def variables(self):
        if self._labels is None or len(self._labels) == 0:
            return []

        return list(self._labels.keys())

    @property
    def paths(self):
        return self._paths

    @property
    def filenames(self):
        return [os.path.basename(p) for p in self.paths]

    @property
    def ids(self):
        return [f.split('.')[0] for f in self.filenames]

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value: str):
        valid = self.variables + [None, 'path', 'filename', 'id']
        if value not in valid:
            raise ValueError((f'Unable to set target {value}. '
                              f'Must be in {valid}'))

        self._target = value
        
    @property
    def y(self):
        if self.target is None:
            return np.asarray([None] * len(self))
        elif self.target == 'path':
            return self.paths
        elif self.target == 'filename':
            return self.filenames
        elif self.target == 'id':
            return self.ids

        return self._labels[self.target]
    
    
    def __init__(self, paths: np.ndarray, 
                 labels: Dict[str, np.ndarray] = None, 
                 target: str = None) -> NiftiDataset:
        self._paths = paths
        self._labels = labels
        self.target = target

    def __len__(self) -> int:
        return len(self.paths)
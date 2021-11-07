from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd

from typing import Any, Dict, List

from .multi_label_dataset import MultiLabelDataset
from ...utils.decorators import json_serialized_property


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

class NiftiDataset(MultiLabelDataset):
    @classmethod
    def from_folder(cls, root: str, *, images: str = 'images', 
                    labels: str = 'labels.csv', suffix: str = 'nii.gz', 
                    **kwargs) -> NiftiDataset:
        images = os.path.join(root, images)
        labels = os.path.join(root, labels)

        df = pd.read_csv(labels, index_col=None)

        assert 'id' in df.columns, 'labels is missing id column'

        label_ids = set(df['id'].astype(str))
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
    def paths(self) -> np.ndarray:
        return self._paths

    @property
    def filenames(self) -> np.ndarray:
        return np.asarray([os.path.basename(p) for p in self.paths])

    @property
    def ids(self) -> np.ndarray:
        return np.asarray([f.split('.')[0] for f in self.filenames])

    @property
    def targets(self) -> List[Any]:
        return super().targets + ['path', 'filename', 'id']
        
    @property
    def y(self) -> np.ndarray: 
        if self.target == 'path':
            return self.paths
        elif self.target == 'filename':
            return self.filenames
        elif self.target == 'id':
            return self.ids
        else:
            return super().y

    @json_serialized_property
    def json(self) -> str:
        obj = super().json

        obj['paths'] = self.paths

        return obj
    
    def __init__(self, paths: np.ndarray, 
                 labels: Dict[str, np.ndarray] = None, 
                 target: str = None) -> NiftiDataset:
        self._paths = paths if isinstance(paths, np.ndarray) \
                      else np.asarray(paths)

        super().__init__(labels, target)

    def __len__(self) -> int:
        return len(self.paths)

    def __eq__(self, other: NiftiDataset) -> bool:
        if not isinstance(other, NiftiDataset):
            return False

        return super().__eq__(other) and \
               np.array_equal(self.paths, other.paths)

    def __add__(self, other: NiftiDataset) -> NiftiDataset:
        assert isinstance(other, NiftiDataset), \
            f'Unable to add NiftiDataset with {type(other)}'

        paths = np.concatenate([self.paths, other.paths])

        labels = {}

        if self.labels is not None:
            for key in self.labels:
                other_values = other.labels[key] if other.labels is not None \
                                                    and key in other.labels \
                               else np.repeat(np.nan, len(other)) 
                labels[key] = np.concatenate([self.labels[key], 
                                              other_values])

        if other.labels is not None:
            for key in other.labels:
                self_values = self.labels[key] if self.labels is not None \
                                                    and key in self.labels \
                               else np.repeat(np.nan, len(other)) 
                labels[key] = np.concatenate([self_values, 
                                              other.labels[key]])

        labels = None if len(labels) == 0 else labels
        target = None

        if self.target == other.target:
            target = self.target
        elif not self.target == other.target == None:
            logger.warning(('Unable to inherit target from two NiftiDatasets '
                            f'with different targets {self.target} and '
                            f'{other.target}. Resorting to None'))

        return NiftiDataset(paths, labels, target=target)
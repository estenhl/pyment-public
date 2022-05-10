from __future__ import annotations
from ast import Mult

import logging
import os
import numpy as np
import pandas as pd

from enum import Enum
from functools import reduce
from typing import Any, Dict, List, Set, Union
from pyment.utils.io.json import encode_object_as_json

from .nifti_dataset import NiftiDataset
from ...utils.decorators import json_serialized_property
from ...utils.io.json import encode_object_as_json


LOGFORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

class MultisampleNiftiDataset(NiftiDataset):
    """Represents a NiftiDataset which has multiple entries per subject.
    Importantly, the high level interface does _NOT_ give access to all
    samples from a given subject. Rather, based on the given strategy,
    a single sample from each subject is used."""

    class Strategy(Enum):
        FIRST = 'first'
        LAST = 'last'
        RANDOM = 'random'

    @classmethod
    def from_folder(cls, root: str, *, images: str = 'images',
                    labels: str = 'labels.csv', suffix: str = 'nii.gz',
                    **kwargs) -> MultisampleNiftiDataset:
        images = os.path.join(root, images)
        labels = os.path.join(root, labels)

        df = pd.read_csv(labels, index_col=None, dtype={'id': 'str'})

        assert 'id' in df.columns, \
            (f'{labels} should contain a field named \'id\' which contains '
            'the image ids used in the dataset')
        assert 'subject' in df.columns, \
            (f'When instantiating a multisample dataset, {labels} should '
             'contain a field named \'subject\' which contains the subject '
             'id for all images')
        if 'path' in df.columns:
            logger.warning((f'{labels} should not contain a field named '
                            '\'path\' as this is used internally. This column '
                            'will be dropped'))

        image_ids = set([filename.split('.')[0] \
                         for filename in os.listdir(images)])
        subjects = df['subject'].values

        return cls.from_image_ids_and_df(image_ids=image_ids, df=df,
                                         image_folder=images, suffix=suffix,
                                         subjects=subjects, **kwargs)

    @property
    def subjects(self) -> Set:
        subjects, idx = np.unique(self._subjects, return_index=True)
        return subjects[np.argsort(idx)]

    @property
    def paths(self) -> np.ndarray:
        """Returns the file paths of the dataset. Returns a single path per
        subject, according to the given strategy."""
        paths = super().paths
        selected_paths = []

        for subject in self.subjects:
            idx = np.where(self._subjects == subject)[0]

            if self.strategy == MultisampleNiftiDataset.Strategy.FIRST:
                idx = idx[0]
            elif self.strategy == MultisampleNiftiDataset.Strategy.LAST:
                idx = idx[-1]
            elif self.strategy == MultisampleNiftiDataset.Strategy.RANDOM:
                idx = np.random.choice(idx)

            selected_paths.append(paths[idx])

        return np.asarray(selected_paths)

    @property
    def y(self) -> np.ndarray:
        """Returns the file paths of the dataset. Returns a single path per
        subject, according to the given strategy."""
        if self.target is None:
            return [None] * len(self)

        y = super().y
        selected_y = []

        for subject in self.subjects:
            idx = np.where(self._subjects == subject)[0]

            if self.strategy == MultisampleNiftiDataset.Strategy.FIRST:
                idx = idx[0]
            elif self.strategy == MultisampleNiftiDataset.Strategy.LAST:
                idx = idx[-1]
            elif self.strategy == MultisampleNiftiDataset.Strategy.RANDOM:
                if len(set(y[idx])) > 1:
                    raise NotImplementedError(('Unable to use '
                                               'multisample-dataset '
                                               'with random sampling '
                                               'and different labels for '
                                               'the same subject'))
                idx = np.random.choice(idx)

            selected_y.append(y[idx])

        return np.asarray(selected_y)

    @json_serialized_property
    def json(self) -> str:
        obj = super().json

        obj['subjects'] = self._subjects
        obj['strategy'] = self.strategy.value

        return obj

    @property
    def _inheritable_keyword_arguments(self) -> Dict[str, Any]:
        data = super()._inheritable_keyword_arguments

        return {key: data[key] for key in data \
                if key != 'subjects'}

    @property
    def shuffled(self):
        subjects = np.random.permutation(self.subjects)
        idx = [np.where(self._subjects == subject)[0] for subject in subjects]
        idx = reduce(lambda x, y: np.concatenate([x, y]), idx)

        return self[idx]

    def __init__(self, *args, subjects: List[str],
                 strategy: Union[str, Strategy] = 'last', **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(strategy, str):
            strategy = MultisampleNiftiDataset.Strategy(strategy)

        self.strategy = strategy
        self._subjects = subjects if isinstance(subjects, np.ndarray) \
                         else np.asarray(subjects)

    def stratified_folds(self, k: int,
                         variables: List[str]) -> MultisampleNiftiDataset:
        raise NotImplementedError(('Stratification with longitudinal data is '
                                   'not implemented'))

    def __len__(self) -> int:
        return len(self.subjects)

    def __add__(self, other: MultisampleNiftiDataset) \
        -> MultisampleNiftiDataset:
        assert isinstance(other, MultisampleNiftiDataset), \
            f'Unable to add NiftiDataset with {type(other)}'

        paths = np.concatenate([self._paths, other._paths])
        subjects = np.concatenate([self._subjects, other._subjects])

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
                               else np.repeat(np.nan, len(self))
                labels[key] = np.concatenate([self_values,
                                              other.labels[key]])

        labels = None if len(labels) == 0 else labels

        if self.target != other.target and self.target is not None:
            logger.warning(('Unable to inherit target from two NiftiDatasets '
                            f'with different targets {self.target} and '
                            f'{other.target}. Resorting to None'))

        return MultisampleNiftiDataset(paths, labels=labels, subjects=subjects,
                                       **self._inheritable_keyword_arguments)

    def __getitem__(self, idx: Any) -> NiftiDataset:
        if isinstance(idx, np.ndarray) or isinstance(idx, slice):
            paths = self._paths[idx]
            subjects = self._subjects[idx]
            labels = self._slice_labels(idx)

            return self.__class__(paths, labels=labels, subjects=subjects,
                                  **self._inheritable_keyword_arguments)
        else:
            raise NotImplementedError()

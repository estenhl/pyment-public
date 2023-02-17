from __future__ import annotations

import logging
import os
import nibabel as nib
import numpy as np
import pandas as pd

from typing import Any, Dict, List, Tuple

from .multi_label_dataset import MultiLabelDataset
from ...utils.decorators import json_serialized_property


LOGFORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

class NiftiDataset(MultiLabelDataset):
    """Represents a dataset with Nifti images, where the datapoints are
    typically a pair consisting of one Nifti image and a label"""

    @classmethod
    def from_folder(cls, root: str, *, images: str = 'images',
                    labels: str = 'labels.csv', suffix: str = 'nii.gz',
                    **kwargs) -> NiftiDataset:
        images = os.path.join(root, images)
        labels = os.path.join(root, labels)

        df = pd.read_csv(labels, index_col=None, dtype={'id': 'str'})

        assert 'id' in df.columns, \
            (f'{labels} should contain a field named \'id\' which contains '
            'the image ids used in the dataset')
        if 'path' in df.columns:
            logger.warning((f'{labels} should not contain a field named '
                            '\'path\' as this is used internally. This column '
                            'will be dropped'))

        image_ids = set([filename.split('.')[0] \
                         for filename in os.listdir(images)])

        return cls.from_image_ids_and_df(image_ids=image_ids, df=df,
                                         image_folder=images, suffix=suffix,
                                         **kwargs)

    @classmethod
    def from_image_ids_and_df(cls, *, image_ids: List[str], df: pd.DataFrame,
                              image_folder: str, suffix: str = 'nii.gz',
                              **kwargs) -> NiftiDataset:
        assert 'id' in df.columns, 'labels is missing id column'

        label_ids = set(df['id'].astype(str))

        missing_images = label_ids - image_ids
        missing_labels = image_ids - label_ids
        complete = label_ids & image_ids

        for image_id in missing_images:
            logger.warning(f'Skipping {image_id}: Missing image')

        for label_id in missing_labels:
            logger.warning(f'Skipping {label_id}: Missing labels')

        df = df[df['id'].isin(complete)]



        create_path = lambda x: os.path.join(image_folder, f'{x}.{suffix}')

        df['path'] = df['id'].apply(create_path)

        paths = df['path'].values
        variables = [var for var in df.columns if var not in ['path', 'id']]
        labels = {var: df[var].values for var in variables}

        logger.debug((f'Creating {cls.__name__} with {len(df)} datapoints and '
                      f'labels {variables}'))

        return cls(paths, labels=labels, **kwargs)

    @property
    def paths(self) -> np.ndarray:
        """Returns the file paths of the dataset"""
        return self._paths

    @property
    def filenames(self) -> np.ndarray:
        """Returns the file basenames of the dataset"""
        return np.asarray([os.path.basename(p) for p in self.paths])

    @property
    def ids(self) -> np.ndarray:
        """Returns the ids (typically the filename minus the suffix) of
        the dataset"""
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

        obj['paths'] = self._paths

        return obj

    @property
    def _inheritable_keyword_arguments(self) -> Dict[str, Any]:
        kwargs = self.json

        return {key: kwargs[key] for key in kwargs \
                  if key not in ['labels', 'paths']}

    @property
    def image_size(self) -> Tuple[int, int, int]:
        return nib.load(self.paths[0]).get_fdata().shape

    @property
    def shuffled(self) -> NiftiDataset:
        idx = np.random.permutation(np.arange(len(self)))

        return self[idx]

    def __init__(self, paths: np.ndarray, *,
                 labels: Dict[str, np.ndarray] = None,
                 **kwargs) -> NiftiDataset:
        self._paths = paths if isinstance(paths, np.ndarray) \
                      else np.asarray(paths)

        super().__init__(labels=labels, **kwargs)

    def stratified_folds(self, k: int, variables: List[str]) -> NiftiDataset:
        """Returns a stratified copy of the dataset, using the variables
        given as input for stratification. Each of these variables must
        correspond to a label of the dataset, e.g. occur in
        dataset.labels

        Args:
            k (int): Number of folds to divide the dataset into
            variables (List[str]): (Ordered) stratification variables
        Returns:
            NiftiDataset: The stratified copy
        Raises:
            """

        data = {key: self.labels[key] for key in self.labels}
        data['path'] = self.paths
        df = pd.DataFrame(data)
        df = df.sort_values(variables)
        df['fold'] = np.arange(len(df)) % k
        df = df.sort_values(['fold'] + variables)

        labels = [{col: df.loc[df['fold'] == i, col].values \
                   for col in df.columns if col not in ['path', 'fold']}
                  for i in range(k)]
        paths = [df.loc[df['fold'] == i, 'path'].values \
                 for i in range(k)]

        return [NiftiDataset(paths[i], labels=labels[i],
                             **self._inheritable_keyword_arguments) \
                for i in range(k)]

    def _slice_labels(self, idx: Any) -> Dict[str, np.ndarray]:
        return {key: self.labels[key][idx] for key in self.labels}

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
                               else np.repeat(np.nan, len(self))
                labels[key] = np.concatenate([self_values,
                                              other.labels[key]])

        labels = None if len(labels) == 0 else labels

        if self.target != other.target and self.target is not None:
            logger.warning(('Unable to inherit target from two NiftiDatasets '
                            f'with different targets {self.target} and '
                            f'{other.target}. Resorting to None'))

        return NiftiDataset(paths, labels=labels,
                            **self._inheritable_keyword_arguments)

    def __eq__(self, other: NiftiDataset) -> bool:
        if not isinstance(other, NiftiDataset):
            return False

        return super().__eq__(other) and \
               np.array_equal(self.paths, other.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: Any) -> NiftiDataset:
        if isinstance(idx, np.ndarray) or isinstance(idx, slice):
            paths = self._paths[idx]
            labels = self._slice_labels(idx)

            return self.__class__(paths, labels=labels,
                                  **self._inheritable_keyword_arguments)
        elif isinstance(idx, tuple):
            variable = idx[1]
            idx = idx[0]

            if not variable in self.labels:
                raise ValueError(f'Dataset does not have variable {variable}')

            value = self.labels[variable][idx]

            if variable in self.encoders:
                if isinstance(idx, (int, np.int32, np.int64)):
                    value = np.asarray([value])

                value = self.encoders[variable].transform(value)

                if isinstance(idx, (int, np.int32, np.int64)):
                    value = value[0]

            return value
        elif isinstance(idx, str):
            if idx in self.variables:
                values = self.labels[idx]

                if idx in self.encoders:
                    values = self.encoders[idx].transform(values)

                return values
            else:
                raise ValueError(('Unable to slice a dataset on a variable '
                                  f'it does not have ({idx} is not in '
                                  f'{self.variables})'))
        else:
            raise ValueError(('Unable to slice NiftiDataset with '
                              f'{idx.__class__.__name__}'))

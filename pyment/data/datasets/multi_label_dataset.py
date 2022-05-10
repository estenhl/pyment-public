from __future__ import annotations

import logging
import os
import numpy as np

from typing import Any, Dict, List, Union


from .dataset import Dataset
from ...labels import load_label_from_json, load_label_from_jsonfile, Label
from ...utils.decorators import json_serialized_property
from ...utils.io.json import encode_object_as_json


LOGFORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLabelDataset(Dataset):
    """An interface class for datasets which has many possible labels.
    A typical example would be a dataset which is loaded from csv,
    where several of the columns can be used as the dataset label
    during a program execution"""

    @property
    def variables(self) -> List[str]:
        """Returns a list of all the variables the dataset has, e.g. the
        possible targets of the dataset

        Returns:
            list: A list of variables usable as targets
        """
        if self._labels is None or len(self._labels) == 0:
            return []

        return list(self._labels.keys())

    @property
    def target(self) -> str:
        """Returns the current target of the dataset. The current target
        defines which label is returned in dataset.y

        Returns:
            str: The current target of the dataset
        """
        return self._target

    @property
    def targets(self) -> List[Any]:
        """Returns a list of all possible targets for the dataset

        Returns:
            list: The possible targets of the dataset
        """
        return self.variables + [None]

    @target.setter
    def target(self, value: Any) -> None:
        """Sets the target of the dataset. Can only be set to a value
        which occurs in the dataset.targets list, or a list of
        multiple such. Can be a string, a Label, or path to a jsonfile
        which stores a Label

        Args:
            value (Union[str, Label]): The value which is set as target
        Raises:
            ValueError: If the given value is not a valid target
        """
        if value is None or isinstance(value, str):
            if not value in self.targets:
                raise ValueError(f'Invalid target {value}')
        elif isinstance(value, list):
            for v in value:
                if not v in self.targets:
                    raise ValueError(f'Invalid target {v}')
        else:
            raise ValueError(f'Invalid target data type {type(value)}')

        self._target = value

    @property
    def labels(self) -> Dict[str, np.ndarray]:
        """Returns the labels of the dataset"""
        return self._labels

    @property
    def y(self) -> np.ndarray:
        if isinstance(self.target, list):
            return np.column_stack([self.get_property(target) \
                                    for target in self.target])

        return self.get_property(self.target)

    @json_serialized_property
    def json(self) -> str:
        encoders = {key: encode_object_as_json(self.encoders[key]) \
                    for key in self.encoders}

        return {
            'labels': self.labels,
            'target': self.target,
            'encoders': encoders
        }

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> Label:
        return cls(**obj)

    def __init__(self, labels: Dict[str, np.ndarray] = None, *,
                 target: str = None,
                 encoders: Dict[str, Label] = None) -> MultiLabelDataset:
        if isinstance(labels, dict):
            for key in labels:
                if isinstance(labels[key], list):
                    labels[key] = np.asarray(labels[key])
                elif not isinstance(labels[key], np.ndarray):
                    raise ValueError('Dataset labels must be numpy arrays')
        elif labels is not None:
            raise ValueError(('Dataset labels must be either None or a '
                              'dictionary'))

        self._labels = labels
        self.target = target
        self.encoders = {}

        if encoders is not None:
            for key in encoders:
                self.add_encoder(key, encoders[key])

    def add_encoder(self, key: str,
                    encoder: Union[str, Dict[str, Any], Label]) -> None:
        if isinstance(encoder, Label):
            self.encoders[key] = encoder
        elif isinstance(encoder, dict):
            self.encoders[key] = load_label_from_json(encoder)
        elif isinstance(encoder, str) and os.path.isfile(encoder):
            self.encoders[key] = load_label_from_jsonfile(encoder)
        else:
            raise ValueError(f'Unable to add encoder of type {type(encoder)}')

    def get_property(self, target: str, index: int = None) -> np.ndarray:
        if target is None:
            labels = np.asarray([None] * len(self))
        else:
            labels = self.labels[target]

        if target in self.encoders:
            labels = self.encoders[target].transform(labels)

        if index is not None:
            return labels[index]

        return labels

    def __eq__(self, other: MultiLabelDataset) -> bool:
        if not isinstance(other, MultiLabelDataset):
            return False

        if not np.array_equal(self.variables, other.variables):
            return False

        for var in self.variables:
            if not np.array_equal(self.labels[var], other.labels[var]):
                return False

        if not self.target == other.target:
            return False

        return super().__eq__(other)

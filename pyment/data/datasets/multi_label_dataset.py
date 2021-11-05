from __future__ import annotations

import logging
import numpy as np

from typing import Any, Dict, List, Union

from .dataset import Dataset
from ...labels import load_label_from_json, Label
from ...utils.decorators import json_serialized_property
from ...utils.io.json import encode_object_as_json


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLabelDataset(Dataset):
    @property
    def variables(self) -> List[str]:
        if self._labels is None or len(self._labels) == 0:
            return []

        return list(self._labels.keys())

    @property
    def paths(self) -> np.ndarray:
        return self._paths

    @property
    def target(self) -> str:
        return self._target

    @property
    def targets(self) -> List[Any]:
        return self.variables + [None]

    @target.setter
    def target(self, value: Any) -> None:
        if value is None or isinstance(value, str) or isinstance(value, Label):
            self._validate_and_set_target(value)
        elif isinstance(value, list):
            if len(value) == 0:
                self._target = None
            elif len(value) == 1:
                self._validate_and_set_target(value[0])
            else:
                for v in value:
                    self._validate_target(v)

                self._target = value
        else:
            raise ValueError(f'Invalid target type {type(value)}')

    @property
    def labels(self) -> Dict[str, np.ndarray]:
        return self._labels

    @property
    def y(self) -> np.ndarray:
        print('Getting y')
        if self.target is None:
            return np.asarray([None] * len(self))
        elif isinstance(self.target, list):
            return np.column_stack([self._get_target_vector(key) \
                                    for key in self.target])
        else:
            return self._get_target_vector(self.target)

    @json_serialized_property
    def json(self) -> str:
        target = self.target if not isinstance(self.target, Label) \
                 else encode_object_as_json(self.target, 
                                            include_timestamp=False,
                                            include_user=False)

        return {
            'labels': self.labels,
            'target': target
        }

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> Label:
        if 'target' in obj and isinstance(obj['target'], dict):
            obj['target'] = load_label_from_json(obj['target'])
    
        return cls(**obj)

    def __init__(self, labels: Dict[str, np.ndarray] = None, 
                 target: str = None) -> MultiLabelDataset:
        if isinstance(labels, dict):
            for key in labels:
                if isinstance(labels[key], list):
                    labels[key] = np.asarray(labels[key])
                elif not isinstance(labels[key], np.ndarray):
                    raise ValueError(f'Dataset labels must be numpy arrays')
        elif labels != None:
            raise ValueError(('Dataset labels must be either None or a '
                              'dictionary'))

        self._labels = labels
        self.target = target

    def _validate_target(self, value: Any) -> None:
        if isinstance(value, Label):
            value = value.name

        if value not in self.targets:
            raise ValueError((f'Unable to set target {value}. '
                            f'Must be in {self.targets}'))

    def _validate_and_set_target(self, value: Any) -> None:
        self._validate_target(value)

        self._target = value

    def _get_target_vector(self, target: Union[str, Label]) -> np.ndarray:
        if isinstance(target, str):
            return self._labels[target]
        elif isinstance(target, Label):
            values = self._labels[target.name]

            if target.is_fitted:
                values = target.transform(values)
            else:
                logger.warning((f'Using unfitted {target.__class__.__name__} '
                                'as target'))

            return values

        raise ValueError(('Unable to retrieve target vector for target with '
                          f'type {type(target)}'))

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

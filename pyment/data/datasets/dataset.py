from __future__ import annotations

import json
import numpy as np

from abc import ABC, abstractproperty
from typing import Any, Dict

from ...utils import save_object_as_json


class Dataset(ABC):
    @abstractproperty
    def y(self) -> np.ndarray:
        """Returns the target vector of the dataset"""
        pass

    @abstractproperty
    def json(self) -> Dict[str, Any]:
        """Returns a json representation of the dataset"""
        pass

    @property
    def jsonstring(self) -> str:
        """Returns a json string representing the dataset"""
        return json.dumps(self.json, indent=4)

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> Dataset:
        """Instantiates a Dataset of the given class from the provided 
        object. Properties of the object must match the corresponding
        constructor"""
        return cls(**obj)

    @classmethod
    def from_jsonstring(cls, jsonstring: str) -> Dataset:
        """Instantiates a Dataset from a jsonstring, via the from_json
        function"""
        return cls.from_json(json.loads(jsonstring))

    def save(self, path: str) -> bool:
        save_object_as_json(self, path)

    @abstractproperty
    def __len__(self) -> int:
        """Returns length of the dataset"""
        pass

    @abstractproperty
    def __eq__(self, other: Dataset) -> bool:
        """Check equality between this dataset and another"""
        pass
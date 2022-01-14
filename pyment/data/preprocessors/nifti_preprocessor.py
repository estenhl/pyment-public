"""Contains the class for preprocessing nifti images."""

from __future__ import annotations

import json
import numpy as np

from typing import Any, Dict

from ...utils.decorators import json_serialized_property

class NiftiPreprocessor:
    """Class for preprocessing nifti images."""

    @json_serialized_property
    def json(self) -> Dict[str, Any]:
        """Returns a json-representation of the preprocessor."""
        return {
            'sigma': self.sigma
        }

    @property
    def jsonstring(self) -> str:
        """Returns a string reprensetation of the preprocessor."""
        return json.dumps(self.json, indent=4)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> NiftiPreprocessor:
        """Instantiates a NiftiPreprocessor from a json object."""
        return cls(**data)

    @classmethod
    def from_jsonstring(cls, data: str) -> NiftiPreprocessor:
        """Instantiates a NiftiPreprocessor from a json string."""
        return cls.from_json(json.loads(data))

    @classmethod
    def from_file(cls, path: str) -> NiftiPreprocessor:
        """Instantiates a NiftiPreprocessor from a file."""
        with open(path, 'r') as f:
            data = f.read()

        return cls.from_jsonstring(data)

    def __init__(self, sigma: float = None):
        self.sigma = sigma

    def save(self, path: str) -> bool:
        """Saves a json-representation of the preprocessor to file."""
        with open(path, 'w') as f:
            f.write(self.jsonstring)

        return True

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Applies the preprocessing configured in the object to the
        given data.
        """
        if self.sigma is not None:
            X = X / self.sigma

        return X

    def __str__(self) -> str:
        return f'NiftiPreprocessor({self.jsonstring})'

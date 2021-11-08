from __future__ import annotations

import json
import numpy as np

from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Union

from .missing_strategy import MissingStrategy
from ..utils.io.json import encode_object_as_jsonstring, save_object_as_json, \
                            JSONSerializable
from ..utils.decorators import json_serialized_property


class Label(ABC, JSONSerializable):
    @abstractproperty
    def is_fitted(self) -> bool:
        """Returns true if the variable is fitted and ready for use 
        (e.g. if fit() has been called, or the label was instantiated 
        with a previous fit"""
        return self._fit is not None
    
    @abstractproperty
    def applicable_missing_strategies(self) -> List[MissingStrategy]:
        """Returns a list of all applicable missing strategies for 
        the given object"""
        pass

    @property
    def missing_strategy(self) -> str:
        return self._missing_strategy

    @missing_strategy.setter
    def missing_strategy(self, strategy: Union[str, MissingStrategy]) -> None:
        if isinstance(strategy, str):
            strategy = MissingStrategy(strategy)
    
        if strategy not in self.applicable_missing_strategies:
            raise ValueError(('Illegal strategy for missing values '
                              f'{strategy} for {self.__class__.__name__}'))

        self._missing_strategy = strategy

    @json_serialized_property
    def json(self) -> Dict[str, Any]:
        obj = {
            'name': self.name,
            'missing_strategy': self.missing_strategy
        }

        obj['fit'] = self._fit

        return obj
    
    @property
    def jsonstring(self) -> str:
        """Returns a json string representing the label"""
        return json.dumps(self.json, indent=4)


    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> Label:
        """Instantiates a Label of the given class from the provided 
        object. Properties of the object must match the corresponding
        constructor"""
        return cls(**obj)

    @classmethod
    def from_jsonstring(cls, jsonstring: str) -> Label:
        """Instantiates a Label from a jsonstring, via the from_json
        function"""
        return cls.from_json(json.loads(jsonstring))

    def save(self, path: str) -> bool:
        save_object_as_json(self, path)
    
    def __init__(self, name: str, 
                missing_strategy: MissingStrategy = MissingStrategy.ALLOW,
                fit: Dict[str, Any] = None) -> Label:
        if fit is None:
            fit = {}
        
        self.name = name
        self.missing_strategy = missing_strategy

        self._fit = fit

    def _encode_missing(values: np.ndarray, 
                        strategy: MissingStrategy) -> np.ndarray:
        if strategy == MissingStrategy.ALLOW:
            return values
        else: 
            raise ValueError(f'Illegal strategy for missing values {strategy}')

    @abstractmethod
    def fit(self, values: np.ndarray) -> None:
        """Fits eventual preprocessing steps based on given values.
        This would typically mean finding min and max for a continuous
        variable that will be normalized, finding an encoding table 
        for binary or categorical variables, etc"""
        pass

    @abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transforms a list of values according to the rules set and 
        the previous fit. Will raise an error if the label is not 
        fitted"""
        pass

    @abstractmethod
    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Convenience-function for combining fit and transform in a 
        single call"""
        pass

    @abstractmethod
    def revert(self, values: np.ndarray) -> np.ndarray:
        """Reverts the operations applied in transform, effectively 
        decoding the values back to their original format"""
        pass

    def __eq__(self, other: Label) -> bool:
        if not isinstance(other, Label):
            return False

        return self.name == other.name and \
               self.missing_strategy == other.missing_strategy and \
               self._fit == other._fit

    def __str__(self) -> str:
        return encode_object_as_jsonstring(self)
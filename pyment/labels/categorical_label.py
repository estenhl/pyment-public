"""Contains the class representing categorical labels."""

from __future__ import annotations

import logging
import numpy as np

from enum import Enum
from typing import Any, Dict, List, Union

from pyment.labels.missing_strategy import MissingStrategy

from .label import Label
from .missing_strategy import MissingStrategy
from ..utils.decorators import json_serialized_property

LOGFORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

class CategoricalLabel(Label):
    """Class representing a categorical label."""

    class Encoding(Enum):
        """Enum for the different categorical encoding possibilities"""
        ONEHOT = 'onehot'
        INDEX = 'index'

    @property
    def is_fitted(self) -> bool:
        """A boolean representing whether the label has been fitted or
        not.
        """
        return self.mapping is not None

    @property
    def encoding(self) -> CategoricalLabel.Encoding:
        """Returns the encoding used by the label."""
        return self._encoding

    @property
    def frequencies(self) -> Dict[Any, int]:
        """Returns the frequencies which were seen during fit."""
        if not self.is_fitted:
            return None

        return self._fit['frequencies']

    @property
    def reference(self) -> Any:
        """Returns the reference level (e.g. the most frequent level
        seen during fit.
        """
        if not self.is_fitted:
            return None

        frequencies = self.frequencies
        frequencies = [(key, frequencies[key]) for key in frequencies]
        frequencies = sorted(frequencies)

        return frequencies[-1][0]
    @property
    def applicable_missing_strategies(self) -> List[MissingStrategy]:
        """Returns a list of the missing strategies which is allowed
        for the label.
        """
        return [MissingStrategy.ALLOW, MissingStrategy.REFERENCE_FILL]

    @encoding.setter
    def encoding(self, value: Union[str, CategoricalLabel.Encoding]) -> None:
        if value is None:
            pass
        elif isinstance(value, str):
            value = CategoricalLabel.Encoding(value)
        elif not isinstance(value, CategoricalLabel.Encoding):
            raise ValueError(('Invalid CategoricalLabel encoding value of type '
                              f'{type(value)}'))
        elif not value in set(item for item in CategoricalLabel.Encoding):
            raise ValueError(('Invalid CategoricalLabel.Encoding value '
                              f'{value}'))

        self._encoding = value

    @property
    def mapping(self) -> Dict[Any, Any]:
        """Returns the mapping used by the label. If the label is not
        fitted, returns None.
        """
        if 'mapping' not in self._fit:
            return None

        return self._fit['mapping']

    @json_serialized_property
    def json(self) -> Dict[str, Any]:
        obj = super().json

        obj['encoding'] = self.encoding

        return obj

    @classmethod
    def index_encode(cls, values: np.ndarray, *, mapping: Dict[Any, int],
                     reference: Any,
                     missing_strategy: MissingStrategy) -> np.ndarray:
        encoded = np.repeat(-1, len(values))

        for key in mapping:
            encoded[np.where(values == key)] = mapping[key]

        if missing_strategy == MissingStrategy.REFERENCE_FILL:
            if values.dtype.type is np.str_:
                nan_idx = [x == 'nan' for x in values]
            else:
                nan_idx = np.where(values == -1)
            encoded[nan_idx] = mapping[reference]

        return encoded.astype(np.int32)

    @classmethod
    def onehot_encode(cls, values: np.ndarray, *, mapping: Dict[Any, int],
                      reference: Any,
                      missing_strategy: MissingStrategy) -> np.ndarray:
        index_encoded=cls.index_encode(values, mapping=mapping,
                                       reference=reference,
                                       missing_strategy=missing_strategy)

        max_value = np.amax(list(mapping.values()))
        encoded = np.zeros((len(index_encoded), max_value+1))

        for i in range(len(encoded)):
            encoded[i][index_encoded[i]] = 1

        return encoded

    def __init__(self, name: str,
                 encoding: Union[str, CategoricalLabel.Encoding],
                 mapping: Dict[Any, Any] = None,
                 missing_strategy: MissingStrategy = MissingStrategy.ALLOW,
                 fit: Dict[str, Any] = None) -> CategoricalLabel:
        super().__init__(name, missing_strategy=missing_strategy, fit=fit)

        self.encoding = encoding

        if mapping is not None:
            self._fit['mapping'] = mapping

    def fit(self, values: np.ndarray) -> None:
        values = np.asarray([x for x in values if (isinstance(x, str) and
                                                   x != 'nan') or \
                                                  (not isinstance(x, str) and
                                                   not np.isnan(x))])
        values, counts = np.unique(values, return_counts=True)
        values = sorted(values)

        self._fit['mapping'] = {values[i]: i for i in range(len(values))}
        self._fit['frequencies'] = {values[i]: counts[i] \
                                    for i in range(len(values))}

        logger.info(('Fitted categorical label with mapping %s and '
                     'frequencies %s'), str(self.mapping),
                     str(self.frequencies))

    def transform(self, values: np.ndarray) -> None:
        if not self.is_fitted:
            raise ValueError(('Unable to call revert on an unfitted '
                              'CategoricalLabel'))
        elif self.encoding == CategoricalLabel.Encoding.INDEX:
            return self.index_encode(values, mapping=self.mapping,
                                     reference=self.reference,
                                     missing_strategy=self.missing_strategy)
        elif self.encoding == CategoricalLabel.Encoding.ONEHOT:
            return self.onehot_encode(values, mapping=self.mapping,
                                      reference=self.reference,
                                      missing_strategy=self.missing_strategy)
        else:
            raise ValueError(f'Invalid encoding {self.encoding}')

    def revert(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

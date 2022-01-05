from __future__ import annotations

import logging
import numpy as np

from enum import Enum
from typing import Any, Dict, Union

from pyment.labels.missing_strategy import MissingStrategy

from .label import Label
from .missing_strategy import MissingStrategy

LOGFORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

class CategoricalLabel(Label):
    class Encoding(Enum):
        ONEHOT = 'onehot'
        INDEX = 'index'

    @property
    def is_fitted(self) -> bool:
        return self.mapping is not None

    @property
    def encoding(self) -> CategoricalLabel.Encoding:
        return self._encoding

    @property
    def applicable_missing_strategies(self) -> List[MissingStrategy]:
        return [MissingStrategy.ALLOW]

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
        if not 'mapping' in self._fit:
            return None

        return self._fit['mapping']

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
        raise NotImplementedError()

    def transform(self, values: np.ndarray) -> None:
        raise NotImplementedError()

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def revert(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

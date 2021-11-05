from __future__ import annotations

import logging
import numpy as np

from collections import Counter
from typing import Any, Dict, List, Set, Union

from pyment.labels.missing_strategy import MissingStrategy

from .label import Label
from .missing_strategy import MissingStrategy

logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryLabel(Label):
    @property
    def is_fitted(self) -> bool:
        return 'encoding' in self._fit and \
               'frequencies' in self._fit

    @property
    def encoding(self) -> Dict[Any, int]:
        if 'encoding' not in self._fit:
            raise ValueError(f'Unfitted BinaryLabel does not have an encoding')

        return self._fit['encoding']

    @property
    def frequencies(self) -> Dict[Any, float]:
        if 'frequencies' not in self._fit:
            raise ValueError(f'Unfitted BinaryLabel does not have frequencies')

        return self._fit['frequencies']

    @property
    def missing_strategy(self) -> str:
        return self._missing_strategy

    @missing_strategy.setter
    def missing_strategy(self, strategy: Union[str, MissingStrategy]) -> None:
        if isinstance(strategy, str):
            strategy = MissingStrategy(strategy)
    
        if strategy not in [MissingStrategy.ALLOW, MissingStrategy.CENTRE_FILL,
                            MissingStrategy.MEAN_FILL]:
            raise ValueError(('Illegal strategy for missing values '
                              f'{strategy} for binary label'))

        self._missing_strategy = strategy

    @property
    def mean(self) -> float:
        if not self.is_fitted:
            raise ValueError('The mean of an unfitted binary label is unknown')

        frequencies = self.frequencies
        total = np.sum([self.encoding[key] * frequencies[key] \
                         for key in frequencies])
        count = np.sum(list(frequencies.values()))

        return total / count

    @property
    def json(self) -> Dict[str, Any]:
        obj = super().json

        return obj

    def __init__(self, name: str, allowed: List[Any] = None,
                 encoding: Set[Any, int] = None, 
                 missing_strategy: MissingStrategy = MissingStrategy.ALLOW,
                 fit: Dict[str, Any] = None) -> BinaryLabel:
        super().__init__(name, missing_strategy=missing_strategy, fit=fit)

        if allowed is not None:
            if encoding is not None:
                raise ValueError(f'Use either allowed or encoding')
            
            assert len(allowed) == 2, \
                'List of allowed values for binary label must have length 2'
            values = sorted(list(allowed))
            encoding = {values[i]: i for i in range(len(values))}

        if encoding is not None:
            assert len(encoding) == 2, \
                'List of encoding for binary label must have length 2'

            self._fit['encoding'] = encoding

    def _encode_missing(self, values: np.ndarray, 
                        strategy: MissingStrategy) -> np.ndarray:
        if strategy == MissingStrategy.ALLOW:
            pass
        elif strategy == MissingStrategy.MEAN_FILL:
            values[np.where(np.isnan(values))] = self.mean
        elif strategy == MissingStrategy.CENTRE_FILL:
            values[np.where(np.isnan(values))] = 0.5

        return values

    def fit(self, values: np.ndarray) -> None:
        counts = Counter(values)
        unique = sorted(list(counts.keys()))
    
        logger.info(f'Found values {unique} for binary variable {self.name}')

        if 'encoding' in self._fit:
            encoding = self._fit['encoding']
        else:
            encoding = {unique[i]: i for i in range(len(unique))}

            assert len(encoding) == 2, \
                'Values provided for fitting binary label has >2 levels'

        logger.info((f'Using encoding {encoding} for binary variable '
                     f'{self.name}'))

        frequencies = {key: counts[key] for key in counts if key in encoding}
        total = np.sum(list(frequencies.values()))
        frequencies = {key: frequencies[key] / total for key in frequencies}

        logger.info((f'Found frequencies {frequencies} for binary variable '
                     f'{self.name}'))

        self._fit = {
            'encoding': encoding,
            'frequencies': frequencies
        }

    def transform(self, values: np.ndarray) -> None:
        if not self.is_fitted:
            raise ValueError((f'Unable to call transform on an unfitted '
                              'BinaryLabel'))

        encoded = np.empty(len(values))
        encoded[:] = np.nan

        for key in self.encoding:
            encoded[np.where(values == key)] = self.encoding[key]
        
        encoded = self._encode_missing(encoded, strategy=self.missing_strategy)

        return encoded

    def fit_transform(self, values: np.ndarray) -> None:
        self.fit(values)
        return self.transform(values)

    def __eq__(self, other: BinaryLabel) -> bool:
        if not isinstance(other, BinaryLabel):
            return False

        return super().__eq__(other)
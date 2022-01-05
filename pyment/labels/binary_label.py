from __future__ import annotations

import logging
import numpy as np

from collections import Counter
from typing import Any, Dict, List, Set

from pyment.labels.missing_strategy import MissingStrategy

from .label import Label
from .missing_strategy import MissingStrategy

LOGFORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryLabel(Label):
    @property
    def is_fitted(self) -> bool:
        return 'mapping' in self._fit and \
               'frequencies' in self._fit

    @property
    def mapping(self) -> Dict[Any, int]:
        if 'mapping' not in self._fit:
            raise ValueError('Unfitted BinaryLabel does not have a mapping')

        return self._fit['mapping']

    @property
    def frequencies(self) -> Dict[Any, float]:
        if 'frequencies' not in self._fit:
            raise ValueError('Unfitted BinaryLabel does not have frequencies')

        return self._fit['frequencies']

    @property
    def mean(self) -> float:
        if not self.is_fitted:
            raise ValueError('The mean of an unfitted binary label is unknown')

        frequencies = self.frequencies
        total = np.sum([self.mapping[key] * frequencies[key] \
                         for key in frequencies])
        count = np.sum(list(frequencies.values()))

        return total / count

    @property
    def applicable_missing_strategies(self) -> List[MissingStrategy]:
        return [MissingStrategy.ALLOW, MissingStrategy.CENTRE_FILL,
                MissingStrategy.MEAN_FILL]

    def __init__(self, name: str, allowed: List[Any] = None,
                 mapping: Dict[Any, int] = None,
                 missing_strategy: MissingStrategy = MissingStrategy.ALLOW,
                 fit: Dict[str, Any] = None) -> BinaryLabel:
        super().__init__(name, missing_strategy=missing_strategy, fit=fit)

        # Validate that object is not initialized both with a previous
        # fit and a new configuration
        params = [
            (mapping, None, 'mapping')
        ]

        for var, default, key in params:
            if key in self._fit and var != default:
                raise ValueError(('Unable to instantiate BinaryLabel '
                                  'with a previous fit and non-default '
                                  f'{key}={var}'))

        if allowed is not None:
            if mapping is not None:
                raise ValueError('Use either allowed or mapping')

            assert len(allowed) == 2, \
                'List of allowed values for binary label must have length 2'
            values = sorted(list(allowed))
            mapping = {values[i]: i for i in range(len(values))}

        if mapping is not None:
            assert len(mapping) == 2, \
                'Mapping for binary label must have length 2'

            self._fit['mapping'] = mapping

    def _encode_missing(self, values: np.ndarray,
                        strategy: MissingStrategy) -> np.ndarray:
        if strategy == MissingStrategy.ALLOW:
            pass
        elif strategy == MissingStrategy.MEAN_FILL:
            values[np.where(np.isnan(values))] = self.mean
        elif strategy == MissingStrategy.CENTRE_FILL:
            values[np.where(np.isnan(values))] = 0.5
        else:
            raise ValueError((f'Invalid missing strategy {strategy} for '
                              'BinaryLabel'))

        return values

    def fit(self, values: np.ndarray) -> None:
        counts = Counter(values)
        unique = sorted(list(counts.keys()))

        if 'mapping' in self._fit:
            mapping = self._fit['mapping']
        else:
            mapping = {unique[i]: i for i in range(len(unique))}

            assert len(mapping) == 2, \
                'Values provided for fitting binary label has >2 levels'

        frequencies = {key: counts[key] for key in counts if key in mapping}
        total = np.sum(list(frequencies.values()))
        frequencies = {key: frequencies[key] / total for key in frequencies}

        logger.info((f'Configured binary variable \'{self.name}\' with '
                     f'mapping {mapping} and frequencies {frequencies}'))

        self._fit = {
            'mapping': mapping,
            'frequencies': frequencies
        }

    def transform(self, values: np.ndarray) -> None:
        if not self.is_fitted:
            raise ValueError(('Unable to call transform on an unfitted '
                              'BinaryLabel'))

        encoded = np.empty(len(values))
        encoded[:] = np.nan

        for key in self.mapping:
            encoded[np.where(values == key)] = self.mapping[key]

        encoded = self._encode_missing(encoded, strategy=self.missing_strategy)

        return encoded

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)

    def revert(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __eq__(self, other: BinaryLabel) -> bool:
        if not isinstance(other, BinaryLabel):
            return False

        return super().__eq__(other)

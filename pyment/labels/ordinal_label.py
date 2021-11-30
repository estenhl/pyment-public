import logging
import numpy as np

from collections import Counter
from typing import Any, Dict, List

from .label import Label
from .missing_strategy import MissingStrategy


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

class OrdinalLabel(Label):
    @property
    def is_fitted(self) -> bool:
        return self.ranges is not None

    @property
    def applicable_missing_strategies(self) -> List[MissingStrategy]:
        return [MissingStrategy.ALLOW]

    @property
    def ranges(self) -> Dict[str, Any]:
        return self._fit['ranges']

    @property
    def mapping(self) -> Dict[Any, float]:
        return {key: np.mean(self.ranges[key]) for key in self.ranges}
    
    def __init__(self, name: str, ranges: Dict[str, Any] = None, 
                 standardize: bool = False, mu: float = 0, sigma: float = 1,
                 missing_strategy: MissingStrategy = MissingStrategy.ALLOW,
                 fit: Dict[str, Any] = None):
        super().__init__(name=name, missing_strategy=missing_strategy, fit=fit)

        self.standardize = standardize

        params = [
            (ranges, None, 'ranges')
        ]

        for var, default, key in params:
            if key in self._fit and var != default:
                raise ValueError(('Unable to instantiate OrdinalLabel '
                                  'with a previous fit and non-default '
                                  f'{key}={var}'))

        if not 'mu' in self._fit:
            self._fit['mu'] = mu
        if not 'sigma' in self._fit:
            self._fit['sigma'] = sigma

        if ranges is not None:
            ranges = {k: (min(ranges[k]), max(ranges[k])) \
                           for k in ranges}

            self._fit['ranges'] = ranges

    def fit(self, values: np.ndarray) -> None:
        frequencies = Counter(values)
        frequencies = {x: frequencies[x] for x in frequencies \
                       if x in self.mapping}

        values = self.transform(values)

        self._fit['frequencies'] = frequencies

        if self.standardize:
            self._fit['mu'] = np.nanmean(values)
            self._fit['sigma'] = np.nanstd(values)

        logger.info((f'Configured ordinal variable \'{self.name}\' with '
                     f'{self.mapping} and frequencies {frequencies}'))

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(('Unable to call transform on an unfitted '
                              'OrdinalLabel'))

        encoded = np.empty(len(values))
        encoded[:] = np.nan

        for key in self.mapping:
            encoded[np.where(values == key)] = self.mapping[key]

        encoded = encoded - self._fit['mu']
        encoded = encoded / self._fit['sigma']

        return encoded

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)

    def revert(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
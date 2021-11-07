from __future__ import annotations

import logging
import numpy as np

from typing import Any, Dict, List

from .label import Label
from .missing_strategy import MissingStrategy


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousLabel(Label):
    @property
    def is_fitted(self) -> bool:
        return 'mu' in self._fit and \
               'sigma' in self._fit

    @property
    def floor(self) -> float:
        return self._fit['floor'] if 'floor' in self._fit else None

    @property
    def ceil(self) -> float:
        return self._fit['ceil'] if 'ceil' in self._fit else None

    @property
    def mean(self) -> float:
        if 'mean' not in self._fit:
            logger.warning((f'ContinuousLabel {self.name} does not have a '
                            'known mean'))
            return np.nan

        return self._fit['mean']

    @property
    def stddev(self) -> float:
        if 'stddev' not in self._fit:
            logger.warning((f'ContinuousLabel {self.name} does not have a '
                            'known standard deviation'))
            return np.nan

        return self._fit['stddev']

    @property
    def min(self) -> float:
        if 'min' not in self._fit:
            logger.warning((f'ContinuousLabel {self.name} does not have a '
                            'known minimum value'))

        return self._fit['min']

    @property
    def max(self) -> float:
        if 'max' not in self._fit:
            logger.warning((f'ContinuousLabel {self.name} does not have a '
                            'known maximum value'))

        return self._fit['max']
    
    @property
    def applicable_missing_strategies(self) -> List[MissingStrategy]:
        return [MissingStrategy.ALLOW, MissingStrategy.MEAN_FILL, 
                MissingStrategy.SAMPLE, MissingStrategy.ZERO_FILL]
    
    def __init__(self, name: str,
                 missing_strategy: MissingStrategy = MissingStrategy.ALLOW, 
                 mu: float = 0, sigma: float = 1, floor: float = None, 
                 ceil: float = None, normalize: bool = False, 
                 standardize: bool = False, 
                 fit: Dict[str, Any] = None) -> ContinuousLabel:
        super().__init__(name, missing_strategy=missing_strategy, fit=fit)

        if normalize and (mu != 0 or sigma != 1):
            raise ValueError(('Mu and/or sigma should not be used alongside '
                              'normalize'))
        if standardize and (mu != 0 or sigma != 1):
            raise ValueError(('Mu and/or sigma should not be used alongside '
                              'standardize'))
        if standardize and normalize:
            raise ValueError('Unable to both standardize and normalize')

        self.normalize = normalize
        self.standardize = standardize

        # Validate that object is not initialized both with a previous
        # fit and a new configuration 
        params = [
            (mu, 0, 'mu'),
            (sigma, 1, 'sigma'),
            (floor, None, 'floor'),
            (ceil, None, 'ceil')
        ]

        for var, default, key in params:
            if key in self._fit and var != default:
                raise ValueError(('Unable to instantiate ContinuousLabel '
                                  'with a previous fit and non-default '
                                  f'{key}={var}'))

        if 'mu' not in self._fit:
            self._fit['mu'] = float(mu)
        if 'sigma' not in self._fit:
            self._fit['sigma'] = float(sigma)

        if 'floor' not in self._fit:
            if floor is not None:
                self._fit['floor'] = float(floor)

        if 'ceil' not in self._fit:
            if ceil is not None:
                self._fit['ceil'] = float(ceil)

    def _encode_missing(self, values: np.ndarray, 
                        strategy: MissingStrategy) -> np.ndarray:
        if strategy == MissingStrategy.ALLOW:
            pass
        elif strategy == MissingStrategy.MEAN_FILL:
            mean = self.mean if not np.isnan(self.mean)\
                             else np.nanmean(values)
            values[np.where(np.isnan(values))] = mean
        elif strategy == MissingStrategy.SAMPLE:
            nans = np.where(np.isnan(values))
            mean = self.mean if not np.isnan(self.mean) \
                             else np.nanmean(values)
            stddev = self.stddev if not np.isnan(self.stddev) \
                                 else np.nanstd(values)
            values[nans] = np.random.normal(loc=mean, scale=stddev, 
                                            size=len(nans[0]))
        elif strategy == MissingStrategy.ZERO_FILL:
            values[np.where(np.isnan(values))] = 0
        else:
            raise ValueError((f'Invalid missing strategy {strategy} for '
                              'ContinuousLabel'))

        return values

    def fit(self, values: np.ndarray, transform: bool = False) -> None:
        if all(np.isnan(values)):
            raise ValueError(f'Unable to fit ContinuousLabel on all nans')
        
        if self.normalize:
            self._fit['mu'] = np.nanmin(values)
            self._fit['sigma'] = np.nanmax(values) - np.nanmin(values)
        elif self.standardize:
            self._fit['mu'] = np.nanmean(values)
            self._fit['sigma'] = np.nanstd(values)

        transformed = self.transform(values)
        
        self._fit['mean'] = np.nanmean(transformed)
        self._fit['stddev'] = np.nanstd(transformed)
        self._fit['min'] = np.nanmin(transformed)
        self._fit['max'] = np.nanmax(transformed)

        logger.info((f'Configured continuous label \'{self.name}\' with '
                     f'mean {round(self._fit["mean"], 2)}, '
                     f'stddev {round(self._fit["stddev"], 2)}, '
                     f'min {round(self._fit["min"], 2)} '
                     f'and max {round(self._fit["max"], 2)}'))

        if transform:
            return transformed
        
    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError((f'Unable to call transform on an unfitted '
                              'ContinuousLabel'))

        values = values - self._fit['mu']
        values = values / self._fit['sigma']

        if self.floor is not None:
            values = np.maximum(values, self.floor)

        if self.ceil is not None:
            values = np.minimum(values, self.ceil)

        self._encode_missing(values, self.missing_strategy)

        return values

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values, transform=True)
from __future__ import annotations

import numpy as np

from .label import Label


class ContinuousLabel(Label):
    @property
    def is_fitted(self) -> bool:
        raise NotImplementedError()
    
    def __init__(self, name: str) -> ContinuousLabel:
        super().__init__(name)

    def fit(self, values: np.ndarray) -> None:
        raise NotImplementedError()

    def transform(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
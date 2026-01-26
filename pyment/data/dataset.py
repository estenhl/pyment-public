from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from typing import Any, Callable


logger = logging.getLogger(__name__)

class Dataset:
    @property
    def class_weights(self) -> dict[Any, float]:
        return self._class_weights

    @class_weights.setter
    def class_weights(self, value: Union[str, dict[Any, float]]):
        if value is None:
            pass
        elif value == 'balanced' or isinstance(value, dict):
            unique_values = sorted(
                np.unique(self.labels[self.target].values)
            )
            weights = compute_class_weight(
                value, 
                classes=np.asarray(unique_values), 
                y=self.labels[self.target].values
            )
            self._class_weights = {
                unique_values[i]: weights[i] 
                for i in range(len(unique_values))
            }
            logger.info(
                'Computed balanced class weights: %s', 
                self._class_weights
            )
        else:
            raise ValueError(f'Invalid class_weights value {value}')
    
    def __init__(
        self,
        labels: pd.DataFrame,
        target: str,
        target_encoder: Callable[[Any], int] = None,
        class_weights: Optional[Union[str, dict[Any, float]]] = None
    ):
        assert 'image_path' in labels.columns, (
            'Labels-file must contain an image_path column'
        )
        if target is not None:
            assert target in labels.columns, (
                f'Labels-file must contain the supplied target-column {target}'
            )

        self.labels = labels
        self.target = target
        self.target_encoder = target_encoder
        self.class_weights = class_weights

    def __len__(self) -> int:
        return len(self.labels)

    def __add__(self, other: Dataset) -> Dataset:
        assert self.target == other.target, (
            'Unable to add datasets with different targets'
        )
        return self.__class__(
            pd.concat([self.labels, other.labels]),
            target=self.target,
            target_encoder=self.target_encoder
        )

"""Abstract base class for pyment datasets."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class Dataset:
    """Abstract base for labelled image datasets.

    Provides class-weight computation and concatenation.
    Subclasses must implement ``to_tensorflow_generator``.
    """

    @property
    def class_weights(self) -> dict[Any, float] | None:
        """Per-class weights, or ``None`` if not set."""

        return self._class_weights

    @class_weights.setter
    def class_weights(self, value: str | dict[Any, float] | None):
        if value is None:
            self._class_weights = None
        elif value == 'balanced' or isinstance(value, dict):
            column = self.labels[self.target]
            assert isinstance(column, pd.Series)
            values = column.to_numpy()
            unique_values = sorted(np.unique(values))

            weights = compute_class_weight(
                value, classes=np.asarray(unique_values), y=values
            )
            self._class_weights = {
                unique_values[i]: weights[i] for i in range(len(unique_values))
            }
            logger.info(
                'Computed balanced class weights: %s', self._class_weights
            )
        else:
            raise ValueError(f'Invalid class_weights value {value}')

    def __init__(
        self,
        labels: pd.DataFrame,
        target: str,
        target_encoder: Callable[[Any], int] | None = None,
        class_weights: str | dict[Any, float] | None = None,
    ) -> None:
        """Initialise a dataset from a pandas DataFrame with labels and
        paths.

        Parameters
        ----------
        labels : pd.DataFrame
            Must contain ``image_path`` and ``target`` columns.
        target : str
            Column in ``labels`` to use as the prediction target.
        target_encoder : Callable[[Any], int] | None, optional
            Maps raw label values to integer indices.
        class_weights : str | dict[Any, float] | None, optional
            ``'balanced'`` to auto-compute weights from class
            frequencies, or an explicit ``{class: weight}`` dict.
        """

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

    @abstractmethod
    def to_tensorflow_generator(self, *args, **kwargs) -> tf.data.Dataset:
        """Build a ``tf.data.Dataset`` representing this dataset."""

        pass

    def __len__(self) -> int:
        return len(self.labels)

    def __add__(self, other: Dataset) -> Dataset:
        """Concatenate two datasets. Target must be the same."""

        assert self.target == other.target, (
            'Unable to add datasets with different targets'
        )
        return self.__class__(
            pd.concat([self.labels, other.labels]),
            target=self.target,
            target_encoder=self.target_encoder,
        )

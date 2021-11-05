from __future__ import annotations

import math
import numpy as np

from collections.abc import Iterator
from typing import Any, Callable, Dict, List, Tuple

from ..io import NiftiLoader
from ...callbacks import Resettable


class NiftiGenerator(Iterator, Resettable):
    @property
    def batches(self) -> int:
        return int(math.ceil(len(self) / self.batch_size))

    def __init__(self, dataset, *, loader: Callable[str, np.ndarray] = None,
                 preprocessor: Callable[np.ndarray, np.ndarray] = None,
                 batch_size: int, infinite: bool = False, 
                 shuffle: bool = False, 
                 name: str = 'NiftiGenerator') -> NiftiGenerator:
        if loader is None:
            loader = NiftiLoader()

        if preprocessor is None:
            preprocessor = lambda x: x

        self.dataset = dataset
        self.loader = loader
        self.preprocessor = preprocessor

        self.batch_size = batch_size
        self.infinite = infinite
        self.shuffle = shuffle

        self.name = name

    def get_image(self, idx: int) -> np.ndarray:
        """Returns a single image identified by the given index"""
        if idx > len(self.dataset):
            raise ValueError((f'Index {idx} out of bounds for generator with '
                             f'{len(self.dataset)} data points'))

        path = self.dataset.paths[idx]
        image = self.loader.load(path)
        image = self.preprocessor(image)

        return image

    def get_label(self, idx: int) -> np.ndarray:
        """Returns a single label identified by the given index"""
        if idx > len(self.dataset):
            raise ValueError((f'Index {idx} out of bounds for generator with '
                             f'{len(self.dataset)} data points'))

        return self.dataset.y[idx]

    def get_datapoint(self, idx: int) -> Dict[str, Any]:
        """Returns an image and a label identified by the given index"""
        datapoint = {
            'image': self.get_image(idx),
            'label': self.get_label(idx)
        }

        return datapoint

    def get_batch(self, start: int, end: int) -> Tuple[np.ndarray]:
        """Returns a batch of images and labels, in two separate numpy
        arrays"""
        if end > len(self.dataset):
            raise ValueError((f'End index {i} out of bounds for generator '
                              f'with {len(dataset)} data points'))

        X = []
        y = []

        for i in range(start, end):
            X.append(self.get_image(i))
            y.append(self.get_label(i))

        X = np.asarray(X)
        y = np.asarray(y)

        return X, y

    def _initialize(self) -> None:
        self.index = 0

        if self.shuffle:
            self.dataset = self.dataset.shuffled()

    def reset(self) -> None:
        self._initialize()

    def __iter__(self) -> NiftiGenerator:
        self._initialize()

        return self

    def __next__(self) -> Tuple[np.ndarray]:
        if not hasattr(self, 'index'):
            raise RuntimeError((f'A {self.__class__.__name__} must be '
                                'initialized through the __iter__-function '
                                'before calling __next__'))
        if self.index >= len(self.dataset):
            if not self.infinite:
                raise StopIteration()

            self._initialize()

        start = self.index
        end = min(start + self.batch_size, len(self.dataset))
        batch = self.get_batch(start, end)
        self.index = end

        return batch

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.get_datapoint(i)

    def __len__(self) -> int:
        return len(self.dataset)

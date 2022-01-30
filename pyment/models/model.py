import os
import numpy as np

from abc import abstractproperty
from collections.abc import Iterator
from tensorflow.keras import Model as KerasModel
from typing import Any, Tuple
from tqdm import tqdm

from .model_type import ModelType
from .utils import WeightRepository


class Model(KerasModel):
    @abstractproperty
    def type(self) -> ModelType:
        """Returns the type of the model, as defined by the ModelType
        enum"""
        pass

    @property
    def input_shape(self) -> Tuple[int]:
        return self.layers[0].input.shape

    def __init__(self, *args, weights: str = None, include_top: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)


        if weights is not None:
            if not os.path.isfile(weights):
                weights = WeightRepository.get_path(
                    model=self.__class__.__name__,
                    weights=weights,
                    include_top=include_top)

            self.load_weights(weights)

    def predict(self, data: Any, *, return_labels: bool = False, **kwargs):
        if isinstance(data, Iterator):
            predictions = None
            labels = None

            for batch in tqdm(data, total=data.batches):
                if isinstance(batch, tuple) and len(batch) == 2:
                    X, y = batch
                elif isinstance(batch, np.ndarray):
                    X = batch
                    y = np.asarray([None] * len(batch))

                batch_predictions = self.predict(X, **kwargs)
                predictions = np.concatenate([predictions, batch_predictions]) \
                              if predictions is not None else batch_predictions
                labels = np.concatenate([labels, y]) if labels is not None else y

            if return_labels:
                return predictions, labels

            return predictions
        elif isinstance(data, np.ndarray):
            return super().predict(data, **kwargs)

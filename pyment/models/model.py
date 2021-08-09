import numpy as np

from collections.abc import Iterator
from tensorflow.keras import Model as KerasModel
from typing import Any
from tqdm import tqdm


class Model(KerasModel):
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
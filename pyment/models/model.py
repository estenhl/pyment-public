import numpy as np

from collections.abc import Iterator
from tensorflow.keras import Model as KerasModel
from tqdm import tqdm


class Model(KerasModel):
    def predict(self, generator: Iterator, *, return_labels: bool = False, 
                **kwargs):
        predictions = None
        labels = None

        for batch in tqdm(generator, total=generator.batches):
            if isinstance(batch, Tuple) and len(batch) == 2:
                X, y = batch
            elif isinstance(batch, np.ndarray):
                X = batch
                y = np.asarray([None] * len(batch))

            batch_predictions = super().predict(X, **kwargs)
            predictions = np.concatenate([predictions, batch_predictions]) \
                          if predictions is not None else batch_predictions
            labels = np.concatenate([labels, y]) if labels is not None else y

        if return_labels:
            return predictions, labels

        return predictions
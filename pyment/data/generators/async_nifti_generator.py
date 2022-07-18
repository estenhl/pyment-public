import logging
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import Any, Dict, Tuple

from .nifti_generator import NiftiGenerator


LOGFORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncNiftiGenerator(NiftiGenerator):
    def __init__(self, *args, threads: int,
                 avoid_singular_batches: bool = False, initialize: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if threads < 2:
            raise ValueError(('AsyncNiftiGenerator must have at least 2 '
                              'threads to use'))

        self.threads = threads
        self.avoid_singular_batches = avoid_singular_batches

        self._exception = None
        self._next_batch = None

        self._initialize()

    def reset(self) -> None:
        self._initialize()

    def _initialize(self) -> None:
        logger.debug(f'Initializing AsyncNiftiGenerator {self.name}')

        if hasattr(self, 'threadpool'):
            self.threadpool.shutdown(wait=True)

        self.threadpool = ThreadPoolExecutor(max_workers=self.threads)
        self._next_batch = None

        super()._initialize()

        self._preload()

        logger.debug(f'Finished initializing {self.name}')

    def _preload_datapoint(self, index: int) -> Dict[str, Any]:
        try:
            return self.get_datapoint(index)
        except Exception as e:
            self._exception = e

    def _preload_next_batch(self) -> None:
        start = self.index
        end = min(self.index + self.batch_size, len(self.dataset))
        idx = np.arange(start, end)

        self._preload_next_batch_from_indexes(idx)

    def _preload_next_batch_from_indexes(self, idx: np.ndarray) -> None:
        if len(idx) == 1 and self.avoid_singular_batches:
            idx = np.concatenate([idx, [np.random.randint(len(self))]])

        futures = self.threadpool.map(self._preload_datapoint, idx)
        results = [f for f in futures]
        X = [result['image'] for result in results]
        y = [result['label'] for result in results]

        X = np.asarray(X)
        y = np.asarray(y)

        if len(self.additional_inputs) > 0:
            X = [X] + [np.asarray([datapoint[key] for datapoint in results]) \
                                   for key in self.additional_inputs]

        self._next_batch = X, y

    def _preload(self) -> None:
        if self._next_batch is not None:
            raise RuntimeError(('Unable to start loading next batch before '
                                'the previous batch is fetched'))

        self.threadpool.submit(self._preload_next_batch)

    def release(self) -> None:
        self.threadpool.shutdown(wait=True)

    def __next__(self) -> Tuple[np.ndarray]:
        if self.index >= len(self.dataset):
            raise StopIteration()

        while self._next_batch is None:
            if self._exception:
                raise self._exception

            sleep(0.1)

        X, y = self._next_batch

        if len(self.additional_inputs) > 0:
            assert len(X[0]) == len(y), \
                   'Got different number of images and labels'
        else:
            assert len(X) == len(y), \
                   'Got different number of images and labels'

        self._next_batch = None

        if len(self.additional_inputs) > 0:
            self.index += len(X[0])
        else:
            self.index += len(X)

        if self.index >= len(self.dataset) and self.infinite:
            super()._initialize()

        self._preload()

        if len(y.shape) == 1:
            y = np.reshape(y, (-1, 1))

        return X, y

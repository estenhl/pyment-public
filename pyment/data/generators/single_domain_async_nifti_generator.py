"""Contains the (async) nifti generator which generates batches all from
the same domain.
"""

import logging
import math
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from typing import Dict, List, Tuple

from .async_nifti_generator import AsyncNiftiGenerator


LOGFORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleDomainAsyncNiftiGenerator(AsyncNiftiGenerator):
    """A generator which serves batches containing nifti images in such
    a way that every single batch originates from a single domain."""

    @property
    def batches(self) -> int:
        return len(self._batches)

    def __init__(self, dataset, *args, domain: str,
                 additional_inputs: List[Dict] = None, **kwargs):
        self.domain = domain

        if additional_inputs is None:
            additional_inputs = []

        additional_inputs.append(domain)

        super().__init__(dataset, *args, additional_inputs=additional_inputs,
                         **kwargs)

    def _configure_batches(self) -> None:
        domains = self.dataset[self.domain]
        for domain in np.unique(domains):
            print(domain)
            print(np.where(domains == domain))
            print(np.where(domains == domain)[0])
        indexes = [np.where(domains == domain)[0] \
                   for domain in np.unique(domains)]

        if self.shuffle:
            indexes = [np.random.permutation(index) for index in indexes]

        batches = [np.array_split(indexes[i],
                                  math.ceil(len(indexes[i]) \
                                            / self.batch_size)) \
                   for i in range(len(indexes))]

        batches = reduce(lambda x, y: x + y, batches)

        if self.shuffle:
            idx = np.random.permutation(len(batches))
            batches = [batches[i] for i in idx]

        self._batches = batches

    def _initialize(self) -> None:
        logger.debug('Initializing SingleDomainAsyncNiftiGenerator %s',
                     self.name)

        if hasattr(self, 'threadpool'):
            self.threadpool.shutdown(wait=True)

        self.threadpool = ThreadPoolExecutor(max_workers=self.threads)
        self._next_batch = None
        self.index = 0
        self.batch_index = 0

        self._configure_batches()

        self._preload()

        logger.debug('Finished initializing %s', self.name)

    def _preload_next_batch(self) -> None:
        idx = self._batches[self.batch_index]
        return self._preload_next_batch_from_indexes(idx)

    def __next__(self) -> Tuple[np.ndarray]:
        X, y = super().__next__()

        self.batch_index += 1

        if self.batch_index >= self.batches and self.infinite:
            self._initialize()

        return X, y

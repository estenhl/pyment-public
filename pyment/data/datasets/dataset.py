import numpy as np

from abc import ABC, abstractproperty


class Dataset(ABC):
    @abstractproperty
    def y(self) -> np.ndarray:
        """Returns the target vector of the dataset"""

    def __len__(self):
        return len(self.y)
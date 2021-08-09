import numpy as np

from abc import ABC, abstractproperty


class Dataset(ABC):
    @abstractproperty
    def paths(self) -> np.ndarray:
        """Returns a list of all the paths of the dataset"""
        pass

    @abstractproperty
    def y(self) -> np.ndarray:
        """Returns the target vector of the dataset"""

    def __len__(self):
        return len(self.paths)
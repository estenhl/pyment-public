import numpy as np

from tensorflow.keras import Model as KerasModel

from .weight_repository import WeightRepository
from ..postprocessing import get_postprocessing

class Model(KerasModel):
    def __init__(self, *args, include_top: bool, weights: str, **kwargs):
        super().__init__(*args, **kwargs)
        if weights is not None:
            weights_path = WeightRepository.get_weights(
                classname=self.__class__.__name__,
                include_top=include_top,
                name=weights
            )

            self.load_weights(weights_path)

        self.weight_name = weights
    def postprocess(self, values: np.ndarray) -> np.ndarray:
        f = get_postprocessing(modelname=self.__class__.__name__,
                               weights=self.weight_name)

        return f(values)

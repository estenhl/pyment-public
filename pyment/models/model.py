import os
import numpy as np

from tensorflow.keras import Model as KerasModel

from .weight_repository import WeightRepository
from ..postprocessing import get_postprocessing

class Model(KerasModel):
    """ A model wrapper on top of the default keras Model class.
    Contains two main additions to the standard funcionality: first,
    allows the models implemented here to load weights via the
    ModelRepository. Second, bundles the appropriate postprocessing
    function with each model. """

    def __init__(self, *args, include_top: bool, weights: str, **kwargs):
        super().__init__(*args, **kwargs)

        if weights is not None:
            if os.path.isfile(weights):
                weights_path = weights
            else:
                weights_path = WeightRepository.get_weights(
                    classname=self.__class__.__name__,
                    include_top=include_top,
                    name=weights
                )

            self.load_weights(weights_path)

        self.weight_name = weights

    def postprocess(self, values: np.ndarray) -> np.ndarray:
        """ Applied the appropriate postprocessing to an array of
        predictions. Which postprocessing function should be applied
        is determined with a lookup based on the model class and the
        weights that are used.

        Parameters:
        -----------
        values : np.ndarray
            Raw predictions.

        Returns:
        --------
        np.ndarray
            Processed predictions.
        """

        f = get_postprocessing(modelname=self.__class__.__name__,
                               weights=self.weight_name)

        return f(values)

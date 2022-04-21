"""Contains the class for preprocessing nifti images."""

from __future__ import annotations

import json
import numpy as np

from typing import Any, Dict, List, Tuple

from ...utils.decorators import json_serialized_property


class NiftiAugmenter:
    """Class for augmenting nifti images."""

    @staticmethod
    def flip(image: np.ndarray, flips: List[bool]) -> np.ndarray:
        """Flips a cuboid in all three dimensions.

        Args:
            image (np.ndarray): The image to be flipped
            flips: (List[bool]): A list indicating, for the (y, x, z)
                                 dimension, whether the image should
                                 be flipped in that dimension.

        Returns:
            The flipped image.
        """
        if not np.any(flips):
            return image

        slices = [slice(None, None, -1 if flips[i] else 1) \
                  for i in range(len(flips))]
        image = image[tuple(slices)]

        return image

    @staticmethod
    def zoom(image: np.ndarray, ratios: List[float]) -> np.ndarray:
        return image

    @json_serialized_property
    def json(self) -> Dict[str, Any]:
        """Returns a json-reprensetation of the object."""
        return {
            'flip_probabilities': self.flip_probabilities
        }

    @property
    def jsonstring(self) -> str:
        """Returns a string reprensetation of the NiftiAugmenter."""
        return json.dumps(self.json, indent=4)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> NiftiAugmenter:
        """Instantiates a NiftiAugmenter from json."""
        return cls(**data)

    @classmethod
    def from_jsonstring(cls, data: str) -> NiftiAugmenter:
        """Instantiates a NiftiAugmenter from a jsonstring."""
        return cls.from_json(json.loads(data))

    @classmethod
    def from_file(cls, path: str) -> NiftiAugmenter:
        """Instantiates a NiftiAugmenter from file."""
        with open(path, 'r') as f:
            data = f.read()

        return cls.from_jsonstring(data)

    def __init__(self, flip_probabilities: List[float] = None):
        self.flip_probabilities = flip_probabilities

    def save(self, path: str) -> bool:
        """Saves a json-representation of the augmenter to file."""
        with open(path, 'w') as f:
            f.write(self.jsonstring)

        return True

    def __call__(self, image: np.ndarray) ->  np.ndarray:
        """Applies augmentation to a given image according to the
        parameters set in the object.

        Args:
            image (np.ndarray): The image to augment.

        Returns:
            The augmented image.
        """
        if self.flip_probabilities is not None:
            fp = self.flip_probabilities
            flips = [np.random.uniform(0, 1) < fp[i] for i in range(len(fp))]
            image = self.__class__.flip(image, flips)

        return image

    def __str__(self) -> str:
        return f'NiftiAugmenter({self.jsonstring})'

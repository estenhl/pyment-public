"""Contains the class for preprocessing nifti images."""

from __future__ import annotations

import json
import numpy as np

from scipy.ndimage import affine_transform, zoom
from typing import  Any, Dict, List, Tuple

from ...utils.decorators import json_serialized_property


class NiftiAugmenter:
    """Class for augmenting nifti images."""

    @staticmethod
    def fast_flip(image: np.ndarray, flips: List[bool]) -> np.ndarray:
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
    def fast_shift(image: np.ndarray, shifts: List[int]) -> np.ndarray:
        padding = [tuple(np.maximum(0, [-shifts[i], shifts[i]])) \
                   for i in range(len(shifts))]

        y, x, z = image.shape

        image = image[padding[0][1]:y - padding[0][0], \
                      padding[1][1]:x - padding[1][0], \
                      padding[2][1]:z - padding[2][0]]

        image = np.pad(image, padding, mode='constant', constant_values=0.)

        return image

    @staticmethod
    def flip(image: np.ndarray, flips: List[bool]) -> Tuple[np.ndarray]:
        flip_matrix = np.eye(3) * [-1 if flip else 1 for flip in flips]
        center= np.asarray(image.shape) / 2
        offset = center - center.dot(flip_matrix.T)

        return flip_matrix, offset

    @staticmethod
    def shift(shifts: List[int]) -> Tuple[np.ndarray]:
        return np.eye(3), shifts

    @staticmethod
    def zoom(ratios: List[float]) -> Tuple[np.ndarray]:
        zoom_matrix = np.eye(3) * ratios

        return zoom_matrix, np.zeros(3)

    @staticmethod
    def rotate(image: np.ndarray, rotations: List[int]) -> Tuple[np.ndarray]:
        rotation_matrix = NiftiAugmenter._generate_rotation_matrix(rotations)
        center= np.asarray(image.shape) / 2
        offset=center - center.dot(rotation_matrix.T)


        return rotation_matrix, offset

    @staticmethod
    def shear(shears: List[int]) -> Tuple[np.ndarray]:
        rotation_matrix = np.asarray([
            [1, shears[0], shears[0]],
            [shears[1], 1, shears[1]],
            [shears[2], shears[2], 1]
        ])

        return rotation_matrix, np.zeros(3)

    @staticmethod
    def _generate_rotation_matrix(rotations: np.ndarray) -> np.ndarray:
        rotations = np.radians(rotations)

        rotation_y = np.asarray([
            [1, 0, 0],
            [0, np.cos(rotations[0]), -np.sin(rotations[0])],
            [0, np.sin(rotations[0]), np.cos(rotations[0])]
        ])

        rotation_x = np.asarray([
            [np.cos(rotations[1]), 0, np.sin(rotations[1])],
            [0, 1, 0],
            [-np.sin(rotations[1]), 0, np.cos(rotations[1])]
        ])

        rotation_z = np.asarray([
            [np.cos(rotations[2]), -np.sin(rotations[2]), 0],
            [np.sin(rotations[2]), np.cos(rotations[2]), 0],
            [0, 0, 1]
        ])


        return rotation_y.dot(rotation_x.dot(rotation_z))

    @staticmethod
    def _resolve_probabilities(probabilities: List[float]) -> List[bool]:
        return [np.random.uniform(0, 1) < probabilities[i] \
                for i in range(len(probabilities))]

    @staticmethod
    def _resolve_ranges(ranges: List[Any]) -> List[bool]:
        if all([isinstance(x, int) for x in ranges]):
            return [np.random.randint(-ranges[i], ranges[i]) \
                    for i in range(len(ranges))]

        return [np.random.uniform(-ranges[i], ranges[i]) \
                    for i in range(len(ranges))]

    @json_serialized_property
    def json(self) -> Dict[str, Any]:
        """Returns a json-reprensetation of the object."""
        return {
            'flip_probabilities': self.flip_probabilities,
            'shift_ranges': self.shift_ranges,
            'zoom_ranges': self.zoom_ranges,
            'rotation_ranges': self.rotation_ranges,
            'shear_ranges': self.shear_ranges
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

    def __init__(self, flip_probabilities: List[float] = None,
                 shift_ranges: List[int] = None,
                 zoom_ranges: List[float] = None,
                 rotation_ranges: List[int] = None,
                 shear_ranges: List[int] = None):
        self.flip_probabilities = flip_probabilities
        self.shift_ranges = shift_ranges
        self.zoom_ranges = zoom_ranges
        self.rotation_ranges = rotation_ranges
        self.shear_ranges = shear_ranges

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
        if self.zoom_ranges is None and self.rotation_ranges is None and \
           self.shear_ranges is None:
            if self.flip_probabilities is not None:
                flips = self._resolve_probabilities(self.flip_probabilities)
                image = self.fast_flip(image, flips)

            if self.shift_ranges is not None:
                shifts = self._resolve_ranges(self.shift_ranges)
                image = self.fast_shift(image, shifts)

        else:
            translation_matrix = np.eye(3)
            offset = np.zeros(3)

            if self.flip_probabilities is not None:
                flips = self._resolve_probabilities(self.flip_probabilities)
                flip_matrix, flip_offset = self.flip(image, flips)
                translation_matrix = translation_matrix.dot(flip_matrix)
                offset += flip_offset

            if self.shift_ranges is not None:
                shifts = self._resolve_ranges(self.shift_ranges)
                shift_matrix, shift_offset = \
                    self.shift(shifts)
                translation_matrix = translation_matrix.dot(shift_matrix)
                offset += shift_offset

            if self.zoom_ranges is not None:
                zooms = self._resolve_ranges(self.zoom_ranges)
                zooms = [1 + zoom for zoom in zooms]
                zoom_matrix, zoom_offset = self.zoom(zooms)
                translation_matrix = translation_matrix.dot(zoom_matrix)
                offset += zoom_offset

            if self.rotation_ranges is not None:
                rotations = self._resolve_ranges(self.rotation_ranges)
                rotation_matrix, rotation_offset = \
                    self.rotate(image, rotations)
                translation_matrix = translation_matrix.dot(rotation_matrix)
                offset += rotation_offset

            if self.shear_ranges is not None:
                shears = self._resolve_ranges(self.shear_ranges)
                shear_matrix, shear_offset = \
                    self.shear(shears)
                translation_matrix = translation_matrix.dot(shear_matrix)
                offset += shear_offset

            image = affine_transform(image, translation_matrix, offset=offset, order=2)

        return image

    def __str__(self) -> str:
        return f'NiftiAugmenter({self.jsonstring})'

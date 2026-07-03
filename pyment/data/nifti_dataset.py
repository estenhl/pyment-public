"""Flat-folder dataset of individually preprocessed images."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import ArrayLike

from pyment.loaders.mgh import load_mgh
from pyment.utils.strip_extension import strip_extension

from .dataset import Dataset


class NiftiDataset(Dataset):
    """Dataset backed by a flat folder of individually preprocessed
    images.

    Unlike FastSurferDataset, each subject is a single image file
    directly under a folder (e.g. output from
    preprocess_folder_with_antspynet.py), rather than a FastSurfer
    subject subfolder, and images are not assumed to already share a
    fixed shape.
    """

    @classmethod
    def from_flat_folder(
        cls,
        images_path: str,
        labels_path: str,
        target: str,
        target_encoder: Callable[[Any], ArrayLike] | None = None,
        class_weights: str | dict[Any, float] | None = None,
    ) -> NiftiDataset:
        """Construct a NiftiDataset from a folder of image files.

        Reads files from images_path, strips each filename's
        extension to derive an image_id, and joins them to the
        image_id column in the labels CSV. Rows with no matching file
        or missing target values, and files with no corresponding
        row, are dropped.

        Parameters
        ----------
        images_path : str
            Directory containing one image file per subject.
        labels_path : str
            Path to a CSV with at least image_id and target columns.
        target : str
            Column name to use as the prediction target.
        target_encoder : Callable[[Any], int] | None, optional
            Encoder mapping raw label values to integer indices.
        class_weights : str | dict[Any, float] | None, optional
            Passed through to Dataset.__init__.

        Returns
        -------
        NiftiDataset
        """

        files = {
            strip_extension(filename): os.path.join(images_path, filename)
            for filename in os.listdir(images_path)
            if os.path.isfile(os.path.join(images_path, filename))
        }

        labels = pd.read_csv(labels_path)
        labels['image_path'] = labels['image_id'].map(files)
        labels = labels.dropna(subset=['image_path', target])
        labels = labels.convert_dtypes()

        if isinstance(labels[target].dtype, pd.BooleanDtype):
            labels[target] = labels[target].astype(bool)

        return cls(
            labels=labels,
            target=target,
            target_encoder=target_encoder,
            class_weights=class_weights,
        )

    def to_tensorflow_generator(
        self,
        batch_size: int,
        target_shape: tuple[int, int, int] = (224, 192, 224),
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        """Build a padded, batched tf.data.Dataset.

        Loads each image via load_mgh and pads it up to target_shape
        before batching, so images that don't already share a fixed
        shape (e.g. a minimal brain-bounding-box crop) can still be
        batched together. Padding can only grow a shape, so an image
        exceeding target_shape along any axis will cause batching to
        fail.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        target_shape : tuple[int, int, int], optional
            Shape each image is padded up to before batching.
        shuffle : bool, optional
            Whether to shuffle before batching.

        Returns
        -------
        tf.data.Dataset
            Padded, batched and prefetched (image, target) pairs.
        """

        column = self.labels['image_path']
        assert isinstance(column, pd.Series)
        paths = column.to_list()

        raw_targets = self.labels[self.target].values
        if self.target_encoder:
            targets = np.asarray(
                [self.target_encoder(target) for target in raw_targets]
            )
        else:
            targets = np.asarray(raw_targets)

        dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self))

        dataset = dataset.map(
            lambda image_path, target: (load_mgh(image_path), target),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(target_shape, dataset.element_spec[1].shape),
        )
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

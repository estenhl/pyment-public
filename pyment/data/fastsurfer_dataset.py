"""FastSurfer-organized dataset for SFCN inference."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import ArrayLike

from pyment.loaders.mgh import load_mgh

from .dataset import Dataset
from .utils.ensure_fastsurfer_crops_exists import ensure_fastsurfer_crops_exists

logger = logging.getLogger(__name__)


class FastSurferDataset(Dataset):
    """Dataset backed by FastSurfer subject folders.

    Expects each folder to contain ``mri/orig.mgz`` and
    ``mri/mask.mgz``; lazily generates ``mri/crop.mgz`` on
    first access via ``to_tensorflow_generator``.
    """

    @staticmethod
    def from_paths(
        images_path: str,
        labels_path: str,
        target: str,
        target_encoder: Callable[[Any], ArrayLike] | None = None,
        class_weights: str | dict[Any, float] | None = None,
    ) -> FastSurferDataset:
        """Construct a FastSurferDataset from directory paths.

        Reads subfolders from ``images_path``, joins them to the
        ``image_id`` column in the labels CSV. Rows with no matching
        folder or missing target values and subfolders with no
        corresponding row are dropped.

        Parameters
        ----------
        images_path : str
            Root directory containing one subfolder per subject.
        labels_path : str
            Path to a CSV with at least ``image_id`` and
            ``target`` columns.
        target : str
            Column name to use as the prediction target.
        target_encoder : Callable[[Any], int] | None, optional
            Encoder mapping raw label values to integer indices.
        class_weights : str | dict[Any, float] | None, optional
            Passed through to ``Dataset.__init__``.

        Returns
        -------
        FastSurferDataset
        """

        folders = {
            folder: os.path.join(images_path, folder)
            for folder in os.listdir(images_path)
        }

        labels = pd.read_csv(labels_path)
        labels['image_path'] = labels['image_id'].map(folders)
        labels = labels.dropna(subset=['image_path', target])
        labels = labels.convert_dtypes()

        if isinstance(labels[target].dtype, pd.BooleanDtype):
            labels[target] = labels[target].astype(bool)

        return FastSurferDataset(
            labels=labels,
            target=target,
            target_encoder=target_encoder,
            class_weights=class_weights,
        )

    def to_tensorflow_generator(
        self,
        batch_size: int,
        num_threads: int = 1,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        """Build a batched ``tf.data.Dataset`` for model inference.

        Ensures crops exist for all subjects, then maps each path
        to a loaded MGH volume via ``load_mgh``.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        num_threads : int, optional
            Threads used for parallel crop generation.
        shuffle : bool, optional
            Whether to shuffle before batching.

        Returns
        -------
        tf.data.Dataset
            Batched and prefetched (image, target) pairs.
        """

        column = self.labels['image_path']
        assert isinstance(column, pd.Series)
        values = column.to_list()

        ensure_fastsurfer_crops_exists(values, num_threads=num_threads)

        paths = [
            os.path.join(image_path, 'mri', 'crop.mgz')
            for image_path in self.labels['image_path'].values
        ]

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

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

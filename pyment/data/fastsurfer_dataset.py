from __future__ import annotations

import logging
import nibabel as nib
import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Optional, Union

import tensorflow as tf
from tensorflow_neuroimaging.loaders import load_mgh

from .dataset import Dataset
from .utils.ensure_fastsurfer_crops_exists import (
    ensure_fastsurfer_crops_exists
)
from ..preprocessing.conform import conform


logger = logging.getLogger(__name__)


class FastSurferDataset(Dataset):
    def from_paths(
        images: str,
        labels: str,
        target: str,
        target_encoder: Callable[[Any], int] = None,
        class_weights: Optional[Union[str, dict[Any, float]]] = None
    ) -> FastSurferDataset:
        folders = {
            folder: os.path.join(images, folder)
            for folder in os.listdir(images)
        }

        labels = pd.read_csv(labels)
        labels['image_path'] = labels['image_id'].map(folders)
        labels = labels.dropna(subset=['image_path', target])
        labels = labels.convert_dtypes()

        if isinstance(labels[target].dtype, pd.BooleanDtype):
            labels[target] = labels[target].astype(bool)

        return FastSurferDataset(
            labels=labels,
            target=target,
            target_encoder=target_encoder,
            class_weights=class_weights
        )

    def to_tensorflow_generator(
        self,
        target_shape: tuple[int, int, int],
        batch_size: int,
        num_threads: int = 1,
        shuffle: bool = False
    ) -> tf.data.Dataset:
        ensure_fastsurfer_crops_exists(
            self.labels['image_path'].values,
            num_threads=num_threads,
            target_shape=target_shape
        )

        paths = [
            os.path.join(image_path, 'mri', 'crop.mgz')
            for image_path in self.labels['image_path'].values
        ]

        if self.target_encoder:
            targets = np.asarray(
                [self.target_encoder(target) for target in targets]
            )
        else:
            targets = np.asarray(
                [target for target in self.labels[self.target].values]
            )

        dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self))

        dataset = dataset.map(
            lambda image_path, target: (
                load_mgh(image_path),
                target
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

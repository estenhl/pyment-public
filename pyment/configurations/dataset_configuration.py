"""Pydantic configuration for FastSurfer dataset loading."""

import os
from typing import Any, Callable

from numpy.typing import ArrayLike
from pydantic import BaseModel

from ..data.fastsurfer_dataset import FastSurferDataset


class FastSurferDatasetConfiguration(BaseModel):
    """Configuration for a FastSurfer dataset.

    Specifies paths to the images directory and labels CSV;
    ``build()`` constructs the dataset object.
    """

    images: str
    labels: str

    def build(
        self,
        target: str,
        target_encoder: Callable[[Any], ArrayLike] | None = None,
    ) -> FastSurferDataset:
        """Build a FastSurferDataset from the configured paths.

        Parameters
        ----------
        target : str
            Column name in the labels CSV to use as the target.
        target_encoder : Callable[[Any], int] | None, optional
            Encoder mapping raw label values to integer indices.

        Returns
        -------
        FastSurferDataset
        """

        return FastSurferDataset.from_paths(
            images_path=os.path.expanduser(os.path.expandvars(self.images)),
            labels_path=os.path.expanduser(os.path.expandvars(self.labels)),
            target=target,
            target_encoder=target_encoder,
        )

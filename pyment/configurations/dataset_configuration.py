import os
from typing import Any, Callable

from pydantic import BaseModel

from .target_configuration import BinaryTargetConfiguration
from ..data.fastsurfer_dataset import FastSurferDataset

class FastSurferDatasetConfiguration(BaseModel):
    images: str
    labels: str

    def build(
        self,
        target: str,
        target_encoder: Callable[[Any], int] = None
    ) -> FastSurferDataset:
        images = os.path.expanduser(os.path.expandvars(self.images))

        return FastSurferDataset.from_paths(
            images=images,
            labels=os.path.expanduser(os.path.expandvars(self.labels)),
            target=target,
            target_encoder=target_encoder
        )



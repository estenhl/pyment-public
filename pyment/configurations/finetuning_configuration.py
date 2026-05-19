"""Top-level Pydantic configuration for a finetuning run."""

from pydantic import BaseModel, Field

from .data_split_configuration import DataSplitConfiguration
from .dataset_configuration import FastSurferDatasetConfiguration
from .sfcn_configuration import SFCNConfiguration
from .target_configuration import TargetConfiguration


class FinetuningConfiguration(BaseModel):
    """Full configuration for a pyment finetuning run.

    Validated from a JSON file by ``pyment-finetune``. Each
    sub-configuration has a paired ``build()`` method that
    constructs the corresponding runtime object.
    """

    dataset: FastSurferDatasetConfiguration
    data_split: DataSplitConfiguration
    model: SFCNConfiguration
    target: TargetConfiguration
    pretrained_multitask_weights: str = 'multi-2025'
    batch_size: int
    num_threads: int = 1
    loss: str
    metrics: list[str] = Field(default_factory=list)
    optimizer: str = 'adam'
    epochs: int
    destination: str

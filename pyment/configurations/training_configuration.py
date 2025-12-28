from pydantic import BaseModel, Field

from .dataset_configuration import FastSurferDatasetConfiguration
from .data_split_configuration import DataSplitConfiguration
from .sfcn_configuration import SFCNConfiguration
from .target_configuration import BinaryTargetConfiguration


class TrainingConfiguration(BaseModel):
    dataset: FastSurferDatasetConfiguration
    data_split: DataSplitConfiguration
    model: SFCNConfiguration
    target: BinaryTargetConfiguration
    batch_size: int
    num_threads: int = 1
    loss: str
    metrics: list[str] = Field(default_factory=list)
    optimizer: str = 'adam'
    epochs: int

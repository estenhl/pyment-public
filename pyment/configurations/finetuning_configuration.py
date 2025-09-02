from pydantic import BaseModel

from .dataset_configuration import DatasetConfiguration
from .model_configuration import ModelConfiguration
from .training_configuration import TrainingConfiguration


class FinetuningConfiguration(BaseModel):
    model: ModelConfiguration
    data: DatasetConfiguration
    training: TrainingConfiguration
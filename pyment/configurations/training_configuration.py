from typing import List

from pydantic import BaseModel

from .learning_rate_schedule_configuration import (
    LearningRateScheduleConfiguration
)


class TrainingConfiguration(BaseModel):
    target: str
    loss: str
    metrics: List[str] = None
    optimizer: str
    learning_rate: float
    learning_rate_schedule: LearningRateScheduleConfiguration = None
    batch_size: int
    epochs: int
    destination: str = None
from abc import abstractmethod
from typing import Annotated, Literal, Union

from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from pydantic import BaseModel, ConfigDict, Field


class LearningRateScheduleBaseConfiguration(BaseModel):
    model_config = ConfigDict(extra='forbid')

    @abstractmethod
    def instantiate(self) -> Callback:
        pass

class AnnealingLearningRateScheduleConfiguration(
    LearningRateScheduleBaseConfiguration
):
    kind: Literal['annealing']
    factor: float
    patience: int
    minimum_learning_rate: float

    def instantiate(self) -> Callback:
        return ReduceLROnPlateau(
            factor=self.factor,
            patience=self.patience,
            min_lr=self.minimum_learning_rate,
            verbose=True
        )

class StepWiseLearningRateScheduleConfiguration(
    LearningRateScheduleBaseConfiguration
):
    kind: Literal['stepwise']

    def instantiate(self) -> Callback:
        return ReduceLROnPl

LearningRateScheduleConfiguration = Annotated[
    Union[
        AnnealingLearningRateScheduleConfiguration, 
        StepWiseLearningRateScheduleConfiguration
    ], 
    Field(discriminator='kind')
]
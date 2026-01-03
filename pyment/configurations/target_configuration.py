from abc import abstractmethod, ABC
from typing import Annotated, Any, Callable, Literal, Union

from pydantic import BaseModel, Field


class BaseTargetConfiguration(BaseModel, ABC):
    name: str

class BinaryTargetConfiguration(BaseTargetConfiguration):
    kind: Literal['binary']
    labels: list[Any]

class RegressionTargetConfiguration(BaseTargetConfiguration):
    kind: Literal['regression']

TargetConfiguration = Annotated[
    Union[BinaryTargetConfiguration, RegressionTargetConfiguration],
    Field(discriminator='kind')
]

def compile_target_encoder(
    configuration: BaseTargetConfiguration
) -> Callable[[Any], Any]:
    if (
        isinstance(configuration, BinaryTargetConfiguration) and
        configuration.labels is not None
    ):
        return lambda x: self.labels.index(x)

    return None
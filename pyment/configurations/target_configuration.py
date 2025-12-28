from typing import Any, Callable

from pydantic import BaseModel

class BaseTargetConfiguration(BaseModel):
    name: str

class BinaryTargetConfiguration(BaseTargetConfiguration):
    labels: list[Any]

    def instantiate_encoder(self) -> Callable[[Any], int]:
        return lambda x: self.labels.index(x)

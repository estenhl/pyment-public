from abc import abstractmethod,ABC
from pydantic import BaseModel
from typing import Literal, Union

from ..models.sfcn import BinarySFCN, SFCN

class BaseSFCNConfiguration(BaseModel, ABC):
    input_shape: tuple[int, int, int] = SFCN.DEFAULT_INPUT_SHAPE

    @abstractmethod
    def instantiate(self) -> SFCN:
        pass

class BinarySFCNConfiguration(BaseSFCNConfiguration):
    kind: Literal['sfcn-bin']

    def instantiate(self) -> BinarySFCN:
        return BinarySFCN(input_shape=self.input_shape)

SFCNConfiguration = BinarySFCNConfiguration

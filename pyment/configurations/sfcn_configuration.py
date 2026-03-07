from abc import abstractmethod,ABC
from pydantic import BaseModel
from typing import Annotated, ClassVar, Literal, Optional, Union
from pydantic import Field

from ..models.sfcn import BinarySFCN, RegressionSFCN, SFCN

class BaseSFCNConfiguration(BaseModel, ABC):
    input_shape: tuple[int, int, int] = SFCN.DEFAULT_INPUT_SHAPE
    dropout: float = 0.0
    weights: Optional[str] = None

    cls: ClassVar[type[BinarySFCN]]

    def build(self) -> SFCN:
        return self.cls(
            input_shape=self.input_shape,
            dropout=self.dropout,
            weights=self.weights
        )

class BinarySFCNConfiguration(BaseSFCNConfiguration):
    kind: Literal['sfcn-bin']

    cls = BinarySFCN

class RegressionSFCNConfiguration(BaseSFCNConfiguration):
    kind: Literal['sfcn-reg']

    cls = RegressionSFCN

SFCNConfiguration = Annotated[
    Union[BinarySFCNConfiguration, RegressionSFCNConfiguration],
    Field(discriminator='kind')
]

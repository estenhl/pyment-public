"""Pydantic configurations for SFCN model variants."""

from abc import ABC
from typing import Annotated, ClassVar, Literal, cast

from pydantic import BaseModel, Field

from ..models.sfcn import SFCN, BinarySFCN, RegressionSFCN


class BaseSFCNConfiguration(BaseModel, ABC):
    """Shared configuration for all SFCN variants.

    Subclasses must set ``cls`` to the concrete model class to
    instantiate.
    """

    input_shape: tuple[int, int, int] = SFCN.DEFAULT_INPUT_SHAPE
    dropout: float = 0.0
    weights: str | None = None

    cls: ClassVar[type[SFCN]]

    def build(self) -> SFCN:
        """Instantiate the SFCN model from configuration.

        Returns
        -------
        An SFCN-variant, defined by the class
        """

        object = self.cls(
            input_shape=self.input_shape,
            dropout=self.dropout,
            weights=self.weights,
        )

        return cast(SFCN, object)


class BinarySFCNConfiguration(BaseSFCNConfiguration):
    """Configuration for a binary-output SFCN (``sfcn-bin``)."""

    kind: Literal['sfcn-bin']

    cls = BinarySFCN


class RegressionSFCNConfiguration(BaseSFCNConfiguration):
    """Configuration for a regression-output SFCN (``sfcn-reg``)."""

    kind: Literal['sfcn-reg']

    cls = RegressionSFCN


SFCNConfiguration = Annotated[
    BinarySFCNConfiguration | RegressionSFCNConfiguration,
    Field(discriminator='kind'),
]

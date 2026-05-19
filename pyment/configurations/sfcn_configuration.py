"""Pydantic configurations for SFCN model variants."""

from abc import ABC
from typing import Annotated, Any, ClassVar, Literal, cast

from pydantic import BaseModel, Field

from ..models.sfcn import SFCN, BinarySFCN, RegressionSFCN


class BaseSFCNConfiguration(BaseModel, ABC):
    """Shared configuration for all SFCN variants.

    Subclasses must set ``cls`` to the concrete model class to
    instantiate.
    """

    input_shape: tuple[int, int, int] | None = None
    dropout: float = 0.0

    cls: ClassVar[type[SFCN]]

    def build(self) -> SFCN:
        """Instantiate the SFCN model from configuration.

        Returns
        -------
        An SFCN-variant, defined by the class
        """

        kwargs: dict[str, Any] = {'dropout': self.dropout}
        if self.input_shape is not None:
            kwargs['input_shape'] = self.input_shape

        return cast(SFCN, self.cls(**kwargs))


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

"""Pydantic configurations for prediction targets."""

from abc import ABC
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, Field


class BaseTargetConfiguration(BaseModel, ABC):
    """Base configuration for a prediction target."""

    name: str


class BinaryTargetConfiguration(BaseTargetConfiguration):
    """Configuration for a binary classification target."""

    kind: Literal['binary']
    labels: list[Any]


class RegressionTargetConfiguration(BaseTargetConfiguration):
    """Configuration for a regression target."""

    kind: Literal['regression']


TargetConfiguration = Annotated[
    BinaryTargetConfiguration | RegressionTargetConfiguration,
    Field(discriminator='kind'),
]


def compile_target_encoder(
    configuration: BaseTargetConfiguration,
) -> Callable[[Any], int] | None:
    """Return a label encoder for binary targets, or ``None``.

    For ``BinaryTargetConfiguration``, returns a callable that
    maps a raw label to its integer index in
    ``configuration.labels``. Returns ``None`` for regression
    targets.

    Parameters
    ----------
    configuration : BaseTargetConfiguration
        A validated target configuration.

    Returns
    -------
    Callable[[Any], int] | None
    """
    if (
        isinstance(configuration, BinaryTargetConfiguration)
        and configuration.labels is not None
    ):
        return lambda x: configuration.labels.index(x)

    return None

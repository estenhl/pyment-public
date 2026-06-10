"""Pydantic configurations for prediction targets."""

from abc import ABC
from typing import Annotated, Any, Callable, Literal

import numpy as np
from numpy.typing import ArrayLike
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


class CategoricalTargetConfiguration(BaseTargetConfiguration):
    """Configuration for a multi-class classification target."""

    kind: Literal['categorical']
    labels: list[Any]


TargetConfiguration = Annotated[
    BinaryTargetConfiguration
    | RegressionTargetConfiguration
    | CategoricalTargetConfiguration,
    Field(discriminator='kind'),
]


def compile_target_encoder(
    configuration: BaseTargetConfiguration,
) -> Callable[[Any], ArrayLike] | None:
    """Return a label encoder for classification targets, or ``None``.

    For ``BinaryTargetConfiguration``, returns a callable that maps a
    raw label to its integer index in ``configuration.labels``. For
    ``CategoricalTargetConfiguration``, returns a callable that maps a
    raw label to a one-hot vector. Returns ``None`` for regression
    targets.

    Parameters
    ----------
    configuration : BaseTargetConfiguration
        A validated target configuration.

    Returns
    -------
    Callable[[Any], ArrayLike] | None
    """
    if isinstance(configuration, BinaryTargetConfiguration):
        return lambda x: configuration.labels.index(x)

    if isinstance(configuration, CategoricalTargetConfiguration):
        n = len(configuration.labels)
        return lambda x: np.eye(n)[configuration.labels.index(x)]

    return None

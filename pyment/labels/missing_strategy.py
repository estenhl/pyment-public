"""Contains the enum for missing strategies."""

from enum import Enum


class MissingStrategy(Enum):
    """Enum containing the missing strategies used by labels."""
    ALLOW = 'allow'
    CENTRE_FILL = 'centre_fill'
    MEAN_FILL = 'mean_fill'
    REFERENCE_FILL = 'reference_fill'
    SAMPLE = 'sample'
    ZERO_FILL = 'zero_fill'

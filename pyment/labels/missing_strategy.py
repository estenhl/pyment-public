from enum import Enum


class MissingStrategy(Enum):
    ALLOW = 'allow'
    CENTRE_FILL = 'centre_fill'
    MEAN_FILL = 'mean_fill'

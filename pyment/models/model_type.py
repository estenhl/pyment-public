from enum import Enum


class ModelType(Enum):
    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'
    RANKING = 'ranking'
from typing import List

from pydantic import BaseModel


class DataSplitConfiguration(BaseModel):
    training_fraction: float
    stratification: List[str] = None
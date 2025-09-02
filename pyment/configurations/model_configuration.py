from typing import Any, Dict

from pydantic import BaseModel, Field


class ModelConfiguration(BaseModel):
    type: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    weights: str = None
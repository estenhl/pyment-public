import pandas as pd
from typing import Any, Callable

class Dataset:
    def __init__(
        self,
        labels: pd.DataFrame,
        target: str,
        target_encoder: Callable[[Any], int] = None
    ):
        assert 'image_path' in labels.columns, (
            'Labels-file must contain an image_path column'
        )

        self.labels = labels
        self.target = target
        self.target_encoder = target_encoder

    def __len__(self) -> int:
        return len(self.labels)

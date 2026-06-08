"""Pydantic configuration for train/validation splitting."""

from pydantic import BaseModel

from ..data.dataset import Dataset


class DataSplitConfiguration(BaseModel):
    """Configuration for splitting a dataset into train/validation."""

    training_fraction: float

    def split(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Split a dataset into random training and validation subsets.

        Parameters
        ----------
        dataset : Dataset
            The full dataset to split.

        Returns
        -------
        tuple[Dataset, Dataset]
            Training and validation subsets, split at
            ``training_fraction * len(dataset)``.
        """
        return dataset.random_split(self.training_fraction)

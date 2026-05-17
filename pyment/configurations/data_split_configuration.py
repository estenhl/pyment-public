"""Pydantic configuration for train/validation splitting."""

from pydantic import BaseModel

from ..data.dataset import Dataset


class DataSplitConfiguration(BaseModel):
    """Configuration for splitting a dataset into train/validation."""

    training_fraction: float

    def split(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Split a dataset into training and validation subsets.

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
        train_len = int(self.training_fraction * len(dataset))

        train_labels = dataset.labels.iloc[:train_len]
        validation_labels = dataset.labels.iloc[train_len:]

        return (
            dataset.__class__(
                labels=train_labels,
                target=dataset.target,
                target_encoder=dataset.target_encoder,
            ),
            dataset.__class__(
                labels=validation_labels,
                target=dataset.target,
                target_encoder=dataset.target_encoder,
            ),
        )

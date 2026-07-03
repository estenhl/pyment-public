"""Dataset classes for loading MRI data."""

from .fastsurfer_dataset import FastSurferDataset
from .nifti_dataset import NiftiDataset

__all__ = ['FastSurferDataset', 'NiftiDataset']

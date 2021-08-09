import nibabel as nib
import numpy as np


class NiftiLoader(object):
    def __init__(self):
        self._cache = {}

    def _load(self, path: str) -> nib.Nifti1Image:
        return nib.load(path)

    def load(self, path: str) -> np.ndarray:
        return self._load(path).get_fdata()

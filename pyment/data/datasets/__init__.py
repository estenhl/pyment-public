import json
import sys

from .dataset import Dataset
from .nifti_dataset import NiftiDataset

def load_dataset_from_jsonfile(path: str) -> Dataset:
    with open(path, 'r') as f:
        data = json.load(f)

    classname = data['cls']
    object = data['object']

    try:
        cls = getattr(sys.modules[__name__], classname)
    except AttributeError:
        raise ValueError(f'Unknown Dataset-class {classname}')

    return cls(**object)
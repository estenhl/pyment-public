import json
import sys

from .binary_label import BinaryLabel
from .continuous_label import ContinuousLabel
from .label import Label
from .missing_strategy import MissingStrategy


_label_types = {
    'binary': BinaryLabel,
    'continuous': ContinuousLabel
}

def label_from_type(type: str, **kwargs) -> Label:
    if not type in _label_types:
        raise ValueError(f'Unknown label type {type}')

    return _label_types[type](**kwargs)

Label.types = list(_label_types.keys())
Label.from_type = label_from_type

def load_label_from_jsonfile(path: str) -> Label:
    with open(path, 'r') as f:
        data = json.load(f)

    classname = data['cls']
    object = data['object']

    try:
        cls = getattr(sys.modules[__name__], classname)
    except AttributeError:
        raise ValueError(f'Unknown Label-class {classname}')

    return cls(**object)
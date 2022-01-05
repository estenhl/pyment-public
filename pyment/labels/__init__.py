import json
import sys

from typing import Any, Dict

from .binary_label import BinaryLabel
from .categorical_label import CategoricalLabel
from .continuous_label import ContinuousLabel
from .label import Label
from .missing_strategy import MissingStrategy
from .ordinal_label import OrdinalLabel


_label_types = {
    'binary': BinaryLabel,
    'categorical': CategoricalLabel,
    'continuous': ContinuousLabel,
    'ordinal': OrdinalLabel
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

    return load_label_from_json(data)

def load_label_from_json(data: Dict[str, Any]) -> Label:
    classname = data['cls']
    object = data['object']

    try:
        cls = getattr(sys.modules[__name__], classname)
    except AttributeError:
        raise ValueError(f'Unknown Label-class {classname}')

    return cls(**object)

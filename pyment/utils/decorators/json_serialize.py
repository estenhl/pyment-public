import numpy as np

from enum import Enum
from typing import Any

from ..io.json import encode_object_as_json, JSONSerializable

_json_safe = [int, str, float]

def _recursive_serialize(obj: Any):
    for dtype in _json_safe:
        if isinstance(obj, dtype):
            return obj

    if obj is None:
        return None
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [_recursive_serialize(x) for x in obj]
    elif isinstance(obj, dict):
        return {_recursive_serialize(key): _recursive_serialize(obj[key]) \
                for key in obj}
    elif isinstance(obj, np.ndarray):
        return [_recursive_serialize(x) for x in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, JSONSerializable):
        return encode_object_as_json(obj, include_timestamp=False, 
                                     include_user=False)
    else:
        raise NotImplementedError(('Unable to serialize object of type '
                                   f'{type(obj)}'))

def json_serialize_object(obj: Any):
    return _recursive_serialize(obj)

def json_serialize(f):
    def wrapper(*args, **kwargs):
        json_obj = f(*args, **kwargs)
        json_obj = json_serialize_object(json_obj)

        return json_obj

    return wrapper

def json_serialized_property(f):
    @property
    def wrapper(*args, **kwargs):
        json_obj = f(*args, **kwargs)
        json_obj = json_serialize_object(json_obj)

        return json_obj

    return wrapper
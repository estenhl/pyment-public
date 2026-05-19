"""Utility for recursively converting objects to JSON-serialisable
Python types."""

import math
from typing import Any

import numpy as np


def json_serialize(obj: Any) -> Any:
    """Recursively converts obj to a JSON-serialisable Python type.

    numpy scalars become int or float, numpy arrays become lists, and
    non-finite floats (nan, inf) become None. All other types are
    returned unchanged.

    Parameters
    ----------
    obj : Any
        The object to serialise.

    Returns
    -------
    Any
        A JSON-safe equivalent of obj.
    """
    if isinstance(obj, dict):
        return {json_serialize(k): json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    elif isinstance(obj, (np.ndarray,)):
        return json_serialize(obj.tolist())
    else:
        return obj

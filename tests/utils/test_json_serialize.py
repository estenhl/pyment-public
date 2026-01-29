import json

import numpy as np
import pytest

from pyment.utils.json_serialize import json_serialize


def test_json_serialize_nan_is_strict_json_safe():
    payload = {"value": np.array([np.nan])}

    serialized = json_serialize(payload)

    # Should not raise under strict JSON rules.
    json.dumps(serialized, allow_nan=False)


def test_json_serialize_numpy_scalars_and_arrays():
    payload = {
        "int": np.int64(5),
        "float": np.float32(3.25),
        "array": np.asarray([1, 2, 3], dtype=np.int32),
    }

    serialized = json_serialize(payload)

    assert serialized == {
        "int": 5,
        "float": 3.25,
        "array": [1, 2, 3],
    }
    json.dumps(serialized, allow_nan=False)


def test_json_serialize_nested_lists_and_dicts():
    payload = {
        "outer": [
            {"inner": np.asarray([[1.0, 2.0], [3.0, 4.0]])},
            {"value": np.float64(1.5)},
        ]
    }

    serialized = json_serialize(payload)

    assert serialized == {
        "outer": [
            {"inner": [[1.0, 2.0], [3.0, 4.0]]},
            {"value": 1.5},
        ]
    }
    json.dumps(serialized, allow_nan=False)


def test_json_serialize_unknown_type_passthrough():
    payload = {"items": set([1, 2, 3])}

    serialized = json_serialize(payload)

    assert serialized["items"] is payload["items"]

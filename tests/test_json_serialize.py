import json
import numpy as np

from pyment.utils.decorators.json_serialize import json_serialize_object


def test_serialize_string():
    serialized = json_serialize_object('test')

    assert isinstance(serialized, str), \
        'JSON serializing a string does not return a string'
    assert 'test' == serialized, \
        'JSON serializing a string returns a different string'

def test_serialize_int():
    serialized = json_serialize_object(3)

    assert isinstance(serialized, int), \
        'JSON serializing an int does not return an int'
    assert 3 == serialized, \
        'JSON serializing an int returns a different int'

def test_serialize_float():
    serialized = json_serialize_object(3.14)

    assert isinstance(serialized, float), \
        'JSON serializing a float does not return a float'
    assert 3.14 == serialized, \
        'JSON serializing a float returns a different float'

def test_serialize_list():
    serialized = json_serialize_object([1., 2., 3.])

    assert isinstance(serialized, list), \
        'JSON serializing a list does not return a list'
    assert [1., 2., 3.] == serialized, \
        'JSON serializing a list returns a different list'

def test_serialize_nested_list():
    serialized = json_serialize_object([[1, 2], [3., 4.]])

    assert isinstance(serialized, list), \
        'JSON serializing a nested list does not return a list'
    assert [[1, 2], [3., 4.]] == serialized, \
        'JSON serializing a nested list returns a different list'

def test_serialize_dict():
    serialized = json_serialize_object({'a': 1., 2: 2.})

    assert isinstance(serialized, dict), \
        'JSON serializing a dict does not return a dict'
    assert {'a': 1., 2: 2.} == serialized, \
        'JSON serializing a dict returns a different dict'

def test_serialize_numpy_int():
    serialized = json_serialize_object(np.int64(3))

    assert isinstance(serialized, int), \
        'JSON serializing a numpy int does not return an int'
    assert 3 == serialized, \
        'JSON serializing a numpy int returns a different int'

    exception = False

    try:
        json.dumps(serialized)
    except Exception:
        exception = True

    assert not exception, 'JSON serialization does not handle numpy ints'

def test_serialize_numpy_float():
    serialized = json_serialize_object(np.float32(3.14))

    assert isinstance(serialized, float), \
        'JSON serializing a numpy float does not return a float'
    # Loses some precision in conversion
    assert abs(3.14 - serialized) < 1e-5, \
        'JSON serializing a numpy float returns a different float'

    exception = False

    try:
        json.dumps(serialized)
    except Exception:
        exception = True

    assert not exception, 'JSON serialization does not handle numpy ints'

def test_serialize_nparray():
    serialized = json_serialize_object(np.asarray([1, 2, 3], dtype=np.int64))

    assert isinstance(serialized, list), \
        'JSON serializing an array does not return a list'
    assert [1., 2., 3] == serialized, \
        'JSON serializing an array returns a different list'
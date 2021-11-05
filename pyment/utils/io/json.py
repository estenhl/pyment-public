import json

from abc import ABC, abstractproperty
from datetime import datetime
from getpass import getuser
from typing import Any, Dict


class JSONSerializable:
    @abstractproperty
    def json(self) -> str:
        """Returns a JSON representation of the object"""
        pass


def save_object_as_json(obj: JSONSerializable, path: str, *, 
                        write_timestamp: bool = True, 
                        write_user: bool = True) -> bool:
    data = encode_object_as_json(obj, include_timestamp=write_timestamp,
                                 include_user=write_user)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    return True

def encode_object_as_json(obj: JSONSerializable, 
                          include_timestamp: bool = True,
                          include_user: bool = True) -> Dict[str, Any]:
    obj = {
        'cls': obj.__class__.__name__,
        'object': obj.json
    }

    if include_timestamp:
        obj['timestamp'] = datetime.now().strftime('%y-%m-%d %H:%M:%S')

    if include_user:
        obj['username'] = getuser()

    return obj

def encode_object_as_jsonstring(data: Dict[str, Any]) -> str:
    obj = encode_object_as_json(data, include_timestamp=False,
                                include_user=False)

    return json.dumps(obj, indent=4)
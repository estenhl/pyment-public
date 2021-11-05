import json

from abc import ABC, abstractproperty
from datetime import datetime
from getpass import getuser


class JSONSerializable:
    @abstractproperty
    def json(self) -> str:
        """Returns a JSON representation of the object"""
        pass


def save_object_as_json(obj: JSONSerializable, path: str, *, 
                        write_timestamp: bool = True, 
                        write_user: bool = True) -> bool:
    obj = {
        'cls': obj.__class__.__name__,
        'object': obj.json
    }

    if write_timestamp:
        obj['timestamp'] = datetime.now().strftime('%y-%m-%d %H:%M:%S')

    if write_user:
        obj['username'] = getuser()

    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

    return True
from __future__ import annotations

import json
import numpy as np

from typing import Any, Dict

from ..decorators import json_serialized_property


class LearningRateSchedule:
    @json_serialized_property
    def json(self) -> Dict[str, Any]:
        return {
            'schedule': self.schedule
        }

    @property
    def jsonstring(self) -> str:
        return json.dumps(self.json, indent=4)

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> LearningRateSchedule:
        if not 'schedule' in obj:
            raise KeyError(('JSON object used for instantiating '
                            'LearningRateSchedule must contain a schedule '
                            'field'))
    
        obj['schedule'] = {int(key): float(obj['schedule'][key]) \
                           for key in obj['schedule']}

        return cls(**obj)

    @classmethod
    def from_jsonstring(cls, obj: str) -> LearningRateSchedule:
        return cls.from_json(json.loads(obj))

    @classmethod
    def from_jsonfile(cls, path: str) -> LearningRateSchedule:
        with open(path, 'r') as f:
            obj = json.load(f)

        return cls.from_json(obj)

    def __init__(self, schedule: Dict[int, float]):
        if 0 not in schedule:
            raise ValueError(('LearningRateSchedule must have a learning rate '
                              'for epoch 0'))

        self.schedule = schedule

    def save(self, path: str) -> bool:
        with open(path, 'w') as f:
            f.write(self.jsonstring)

        return True

    def __call__(self, epoch: int) -> float:
        if epoch < 0:
            raise ValueError('Unable to get learning rate for epoch < 0')

        keys = sorted(list(self.schedule.keys()))

        for i in range(len(keys)):
            if keys[i] > epoch:
                return self.schedule[keys[i-1]]

        return self.schedule[np.amax(keys)]
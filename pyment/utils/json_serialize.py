import numpy as np
from typing import Any

def json_serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {json_serialize(k): json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj
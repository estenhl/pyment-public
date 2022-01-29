from typing import Any, Callable, Dict, List


def assert_exception(f: Callable, *, args: List[Any] = None, 
                     kwargs: Dict[str, Any] = None, 
                     exception: Any = Exception,
                     message: str):
    args = args if args is not None else []
    kwargs = kwargs if kwargs is not None else {}

    raised = False

    try:
        f(*args, **kwargs)
    except exception:
        raised = True

    assert raised, message
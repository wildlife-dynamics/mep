from ecoscope_workflows_core.decorators import task
from typing import Sequence, TypeVar,Iterable,Tuple,Union,List,Iterator,Optional

from typing import Iterable, Tuple, List, Union, Optional

GroupKey = Tuple  # normalized group key as a tuple
V = Union[tuple, list, str, int, float, object]

K = TypeVar("K")  # no Hashable bound
L = TypeVar("L")
R = TypeVar("R")
T = TypeVar("T")

JsonPrimitive = Union[str, int, float, bool, None]

@task
def flatten_tuple(nested: tuple) -> Tuple[JsonPrimitive, ...]:
    """
    Recursively flatten a (possibly deeply nested) tuple into a flat tuple
    of JSON-safe primitives.

    Note:
    - We annotate the input as `tuple` (built-in) to avoid Pydantic trying to
      resolve a recursive type annotation at import time.
    - At runtime we enforce that leaf values are JSON-safe primitives and
      raise TypeError for unsupported leaf types (avoids silently accepting objects).
    """
    # If input is not a tuple then it's an invalid call for this function.
    if not isinstance(nested, tuple):
        raise TypeError("flatten_tuple expects a tuple (possibly nested).")

    flat_list: list[JsonPrimitive] = []

    for item in nested:
        if isinstance(item, tuple):
            flat_list.extend(flatten_tuple(item))
            continue

        # Accept JSON-safe primitives
        if isinstance(item, (str, int, float, bool)) or item is None:
            flat_list.append(item)
            continue

        # Reject everything else explicitly
        raise TypeError(
            f"Unsupported leaf type in nested tuple: {type(item)!r}. "
            "Allowed leaf types: str, int, float, bool, None."
        )

    return tuple(flat_list)

@task
def zip_grouped_by_key(
    left: Sequence[tuple[K, L]],
    right: Sequence[tuple[K, R]],
) -> list[tuple[K, tuple[L, R]]]:
    """
    Zips two grouped sequences by key.

    - Preserves order from `left`
    - Keeps only keys present in both
    - Returns [(key, (left_value, right_value)), ...]
    """
    right_dict: dict[K, R] = dict(right)  # runtime still requires hashable keys
    out: list[tuple[K, tuple[L, R]]] = []
    seen: set[K] = set()

    for k, lv in left:
        if k in seen:
            raise ValueError(f"Duplicate key in 'left': {k!r}")
        seen.add(k)
        if k in right_dict:
            out.append((k, (lv, right_dict[k])))

    return out
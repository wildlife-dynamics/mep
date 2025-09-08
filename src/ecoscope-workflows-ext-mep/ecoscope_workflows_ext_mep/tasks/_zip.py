from typing import Sequence, TypeVar
from ecoscope_workflows_core.decorators import task

K = TypeVar("K")  # no Hashable bound
L = TypeVar("L")
R = TypeVar("R")


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

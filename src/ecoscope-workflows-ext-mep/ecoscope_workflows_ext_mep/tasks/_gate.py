from typing import Annotated

from pydantic import Field

from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.skip import SKIP_SENTINEL, SkipSentinel


@task
def gate_earthranger_client(
    client: Annotated[
        str,
        Field(
            description="Named EarthRanger connection to pass through.", exclude=True
        ),
    ],
    enabled: Annotated[
        bool,
        Field(
            default=False,
            description="Include this optional EarthRanger-dependent branch of the workflow. "
            "When off, the client is replaced with a skip sentinel so this step (and anything "
            "built from its return value) is skipped entirely.",
        ),
    ] = False,
) -> str | SkipSentinel:
    """Pass through a named EarthRanger connection only when enabled; otherwise skip.

    Wire a task-group's boolean toggle to `enabled`, then feed this task's return value into
    a downstream task's `client` parameter to make that whole branch optional on the config
    form. Relies on the workflow's default `any_dependency_skipped` skipif condition to cascade
    the skip to anything depending on that downstream task's output.

    `client` and the return value are plain strings (the named connection, e.g. from
    `set_er_connection`), not `EarthRangerClient` -- that alias carries a `BeforeValidator`
    that coerces a connection-name string into a real client object via `name.lower()`. Every
    downstream task's own `client: EarthRangerClient` parameter already performs that coercion
    itself; doing it here too (or applying it to the skip sentinel) would call `.lower()` on
    an already-resolved client object or on the sentinel, crashing with `AttributeError`.
    """
    return client if enabled else SKIP_SENTINEL

from pydantic import Field
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter._filter import TimeRange

@task
def get_previous_period(
    time_range: Annotated[TimeRange, Field(description="Current time range")]
) -> Annotated[TimeRange, Field(description="Previous period time range")]:
    """
    Calculate the previous period based on the current time range duration.
    
    Example:
        If current period is 2026-01-01 to 2026-02-01 (31 days),
        previous period will be 2025-12-01 to 2026-01-01 (31 days)
    """
    duration = time_range.until - time_range.since
    
    return TimeRange(
        since=time_range.since - duration,
        until=time_range.since,
        timezone=time_range.timezone,
        time_format=time_range.time_format
    )
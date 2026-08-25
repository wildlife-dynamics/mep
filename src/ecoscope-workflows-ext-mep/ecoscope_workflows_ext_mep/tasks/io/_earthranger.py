from pydantic import Field
from wt_registry import register
from typing import Annotated
from ecoscope.platform.connections import EarthRangerClient
from ecoscope.platform.annotations import AdvancedField, AnyDataFrame

@register()
def get_subjects(
    client: EarthRangerClient,
    include_inactive: Annotated[
        bool,
        AdvancedField(default=None, description="Include inactive subjects in the list."),
    ] = None,
    bbox: Annotated[
        tuple[float, float, float, float] | None,
        Field(
            description="Bounding box filter as (west, south, east, north). "
            "Includes subjects with track data inside the box."
        ),
    ] = None,
    subject_group_id: Annotated[str | None, Field(description="Subject group ID to filter subjects by.")] = None,
    subject_group_name: Annotated[str | None, Field(description="Subject group name to filter subjects by.")] = None,
    name: Annotated[str | None, Field(description="Filter subjects by name.")] = None,
    updated_since: Annotated[
        str | None, Field(description="Only include subjects updated since this timestamp (ISO).")
    ] = None,
    updated_until: Annotated[
        str | None, Field(description="Only include subjects updated until this timestamp (ISO).")
    ] = None,
    tracks: Annotated[bool | None, Field(description="Whether to include recent tracks for each subject.")] = None,
    ids: Annotated[
        list[str] | None,
        Field(description="List of subject IDs to fetch. Splits requests in chunks if large."),
    ] = None,
    max_ids_per_request: Annotated[
        int,
        Field(description="Maximum number of IDs per request when splitting batched subject queries."),
    ] = 50,
    raise_on_empty: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Whether to abort the workflow if no subjects are returned from EarthRanger.",
        ),
    ] = True,
) -> AnyDataFrame:
    """Fetch subjects from EarthRanger with filtering options."""

    df = client.get_subjects(
        include_inactive=include_inactive,
        bbox=bbox,
        subject_group_id=subject_group_id,
        subject_group_name=subject_group_name,
        name=name,
        updated_since=updated_since,
        updated_until=updated_until,
        tracks=tracks,
        id=",".join(str(i) for i in ids) if ids is not None else None,
        max_ids_per_request=max_ids_per_request,
    )

    if raise_on_empty and df.empty:
        raise ValueError("No data returned from EarthRanger for get_subjects")
    return df
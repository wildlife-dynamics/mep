from pydantic import Field
from typing import Annotated
from wt_registry import register
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.tasks.filter._filter import TimeRange

ColumnName = Annotated[str, Field(description="Column to aggregate")]


@register()
def dataframe_column_unique(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[list, Field(description="The number of unique values in the column")]:
    return df[column_name].unique()

@register()
def reset_dataframe_index(
    df: AnyDataFrame,
    drop: bool = False,
) -> AnyDataFrame:
    """Reset the index of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input frame.
    drop : bool, default False
        If False, the old index becomes a column (commonly `id` or
        `index`). If True, the old index is discarded entirely.

    Returns
    -------
    DataFrame
        A new frame with the index reset.
    """

    df = df.reset_index(drop=drop)
    return df

@register()
def add_time_since_visit(
    df: AnyDataFrame,
    time_range: TimeRange,
) -> AnyDataFrame:
    """Add elapsed-time columns measured from each row's last visit.

    Computes the gap between `time_range.since` and `last_visited`,
    then writes it out as `hours_since_visit` and `days_since_visit`.

    Parameters
    ----------
    df : DataFrame
        Input frame. Must contain the columns: last_visited.
    time_range : TimeRange
        The reference range; `time_range.since` is the anchor point
        elapsed time is measured from.

    Returns
    -------
    GeoDataFrame
        A copy with `hours_since_visit` and `days_since_visit` added.

    Raises
    ------
    ValueError
        If any required column is missing from `gdf`.
    """
    since = time_range.until - df["last_visited"]
    df["hours_since_visit"] = since.dt.total_seconds() / 3600
    df["days_since_visit"] = df["hours_since_visit"] / 24
    return df

@register()
def add_non_null_flag(
    df: AnyDataFrame,
    source_column: str,
    flag_column: str = "is_present",
) -> AnyDataFrame:
    """Add a boolean column flagging where `source_column` is not null.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input frame.
    source_column : str
        Column to test for non-null values.
    flag_column : str, default "is_present"
        Name of the boolean column to create.

    Returns
    -------
    GeoDataFrame
        A copy with the boolean flag column added.

    Raises
    ------
    ValueError
        If `source_column` is missing from `gdf`.
    """
    df[flag_column] = df[source_column].notna()
    return df
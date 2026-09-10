import pandas as pd
from pydantic import Field
from typing import Annotated
from wt_registry import register
from ..spatial_operations import overlay_gdf
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope.platform.annotations import AnyDataFrame,AnyGeoDataFrame
from ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_join import  spatial_join

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

@register()
def compute_dwell_time(
    patrol_trajectories: AnyGeoDataFrame,
    gridded_spatial_feature: AnyGeoDataFrame,
) -> AnyDataFrame:
    """Compute time spent per grid cell from patrol trajectory segments.

    Moving segments are split at cell boundaries and their duration is
    apportioned by the fraction of segment length in each cell.
    Stationary segments (zero length) contribute their whole duration
    to the containing cell.

    Parameters
    ----------
    patrol_trajectories : GeoDataFrame
        Trajectory segments. Must contain `timespan_seconds`
        and a geometry column.
    gridded_spatial_feature : GeoDataFrame
        Grid cells. Must contain `index` and a geometry column.

    Returns
    -------
    DataFrame
        One row per `index` with `seconds_in_cell`,
        `minutes_in_cell`, and `hours_in_cell`.
    """
    tracks = patrol_trajectories.copy()
    tracks["segment_length"] = tracks.geometry.length

    moving = tracks[tracks["segment_length"] > 0].copy()
    still = tracks[tracks["segment_length"] == 0].copy()

    # moving: split at cell borders, apportion time by length fraction
    pieces = overlay_gdf(
        moving[["timespan_seconds", "segment_length", "geometry"]],
        gridded_spatial_feature[["index", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    pieces["frac"] = pieces.geometry.length / pieces["segment_length"]
    pieces["time_in_cell"] = pieces["timespan_seconds"] * pieces["frac"]

    # stationary: whole duration goes to the containing cell
    if len(still):
        pts = still.copy()
        pts["geometry"] = pts.geometry.representative_point()
        pts = spatial_join(
            pts[["timespan_seconds", "geometry"]],
            gridded_spatial_feature[["index", "geometry"]],
            how="inner",
            predicate="within",
        )
        pts["time_in_cell"] = pts["timespan_seconds"]
        pieces = pd.concat(
            [pieces[["index", "time_in_cell"]], pts[["index", "time_in_cell"]]],
            ignore_index=True,
        )
    else:
        pieces = pieces[["index", "time_in_cell"]]

    # aggregate per cell
    dwell = pieces.groupby("index").agg(seconds_in_cell=("time_in_cell", "sum")).reset_index()
    dwell["minutes_in_cell"] = dwell["seconds_in_cell"] / 60
    dwell["hours_in_cell"] = dwell["seconds_in_cell"] / 3600
    return dwell

@register()
def operational_days(
    trajs: AnyDataFrame,
    groupby_cols: list[str],
    time_range: TimeRange,
) -> AnyDataFrame:
    """Distinct operational days per group from patrol trajectory segments.

    A segment that spans midnight counts each calendar date once.

    Parameters
    ----------
    trajs : DataFrame with one row per patrol segment.
    groupby_cols : columns to group by (e.g. ["subject_id", "patrol_subject"]).
    time_range : TimeRange
        The configured reporting window. `Reporting Period (Days)` is based
        on `time_range.since`/`time_range.until`, not on how much of that
        window the trajectory data actually covers.
    """
    trajs = trajs.copy()
    trajs["segment_start"] = pd.to_datetime(trajs["segment_start"])
    trajs["segment_end"] = pd.to_datetime(trajs["segment_end"])

    start_day = trajs["segment_start"].dt.normalize()
    end_day = trajs["segment_end"].dt.normalize()
    same_day = start_day == end_day

    # fast path: segments that don't cross midnight contribute a single day
    single = trajs.loc[same_day, groupby_cols].copy()
    single["day"] = start_day[same_day]

    # slow path: only segments that actually span multiple calendar days
    # need the per-row date_range expansion
    def days_covered(row):
        return pd.date_range(
            row["segment_start"].normalize(),
            row["segment_end"].normalize(),
            freq="D",
        )

    multi = trajs.loc[~same_day]
    if len(multi):
        exploded_multi = multi.assign(day=multi.apply(days_covered, axis=1)).explode("day")
        exploded = pd.concat([single, exploded_multi[groupby_cols + ["day"]]], ignore_index=True)
    else:
        exploded = single
    print(f"same-day segments: {int(same_day.sum())}, multi-day segments: {int((~same_day).sum())}")

    # distinct days on patrol per group
    op_days = exploded.groupby(groupby_cols)["day"].nunique().rename("days_on_patrol")

    # length of the configured reporting period, independent of how much of
    # it the trajectory data actually covers
    since = pd.Timestamp(time_range.since).normalize()
    until = pd.Timestamp(time_range.until).normalize()
    period_days = (until - since).days + 1

    op_df = op_days.reset_index()
    op_df["reporting_period_days"] = period_days
    op_df["active_days_percentage"] = (op_df["days_on_patrol"] / period_days * 100).round(1)
    print(f"period_days: {period_days}, output shape: {op_df.shape}")
    return op_df

@register()
def compute_patrol_effort_fraction(gdf: AnyGeoDataFrame) -> float:
    """Percentage of the gridded area that has been patrolled (visited).

    Returns a value in [0, 100] = (patrolled area / total area) * 100.
    Expects non-overlapping grid cells in a projected CRS.
    """
    print(f"[compute_patrol_effort_fraction] input shape: {gdf.shape}, CRS: {gdf.crs}")
    # 1. required column
    if "visit_bin" not in gdf.columns:
        raise KeyError("Expected a 'visit_bin' column in the GeoDataFrame.")

    # 2. CRS must be projected, or .area is in square degrees (meaningless)
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS; set one before computing area.")
    if gdf.crs.is_geographic:
        raise ValueError(
            f"CRS {gdf.crs.to_epsg()} is geographic. Reproject to a projected CRS "
            "(e.g. gdf.to_crs(gdf.estimate_utm_crs())) before calling this."
        )

    # 3. guard against an empty / zero-area frame
    total_area = gdf.geometry.area.sum()
    if total_area == 0:
        return 0.0

    patrolled_area = gdf.loc[gdf["visit_bin"] != "Unvisited", "geometry"].area.sum()
    fraction = patrolled_area / total_area
    percentage = round(fraction * 100, 2)
    print(
        f"[compute_patrol_effort_fraction] patrolled area: {patrolled_area:.2f}, "
        f"total area: {total_area:.2f}, coverage: {percentage:.2f}%"
    )
    return percentage
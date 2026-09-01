import pandas as pd 
from wt_registry import register
from typing import Literal, cast
from ecoscope.platform.annotations import AnyDataFrame

@register()
def operational_days(
    trajs: AnyDataFrame,
    segment_start: str,
    segment_end: str,
    groupby_cols: list[str],
) -> AnyDataFrame:
    """Distinct operational days per group from patrol trajectory segments.

    A segment that spans midnight counts each calendar date once.

    Parameters
    ----------
    trajs : DataFrame with one row per patrol segment.
    segment_start, segment_end : column names holding the segment timestamps.
    groupby_cols : columns to group by (e.g. ["extra__subject_id", "extra__patrol_subject"]).
    """
    trajs[segment_start] = pd.to_datetime(trajs[segment_start])
    trajs[segment_end] = pd.to_datetime(trajs[segment_end])

    # expand each segment into one row per calendar day it touches
    def days_covered(row):
        return pd.date_range(
            row[segment_start].normalize(),
            row[segment_end].normalize(),
            freq="D",
        )

    exploded = trajs.assign(day=trajs.apply(days_covered, axis=1)).explode("day")
    op_days = exploded.groupby(groupby_cols)["day"].nunique().rename("Days on Patrol")
    period_days = (trajs[segment_end].max().normalize() - trajs[segment_start].min().normalize()).days + 1

    op_df = op_days.reset_index()
    op_df["Reporting Period (Days)"] = period_days
    op_df["Active Days (%)"] = (op_df["Days on Patrol"] / period_days * 100).round(1)
    return op_df

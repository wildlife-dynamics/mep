import pandas as pd
from pydantic import Field
from typing import Annotated
from wt_registry import register
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope_platform.schemas import RelocationsGDFSchema

@register()
def compute_subject_maturity(
    subjects_df: AnyDataFrame,
    relocations_gdf: RelocationsGDFSchema,
    months_duration: Annotated[
        int,
        Field(description="Duration in months to consider an animal mature.", default=6),
    ] = 6,
):
    df = relocations_gdf[["groupby_col", "fixtime"]].copy()
    df["fixtime"] = pd.to_datetime(df["fixtime"], errors="coerce")
    df = df.dropna(subset=["fixtime"])
    span = df.groupby("groupby_col", dropna=False)["fixtime"].agg(first="min", last="max").reset_index()
    span["mature"] = span["last"] >= (span["first"] + pd.DateOffset(months=months_duration))
    subjects_df = subjects_df.merge(span[["groupby_col", "mature"]], on="groupby_col", how="left")
    subjects_df["mature"] = subjects_df["mature"].fillna(False)
    return subjects_df
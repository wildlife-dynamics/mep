import os
import logging
import pandas as pd
from pathlib import Path
from pydantic import Field
from typing import Union, Annotated, cast
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.tasks import analysis
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.analysis._summary import SummaryParam
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

logger = logging.getLogger(__name__)


@task
def compute_maturity(
    subject_df: AnyDataFrame,
    relocations_gdf: AnyDataFrame,
    months_duration: Annotated[
        int,
        Field(description="Duration in months to consider an animal mature.", default=6),
    ] = 6,
    time_column: Annotated[
        str, Field(description="Name of the datetime column in the DataFrame.", default="fixtime")
    ] = "fixtime",
):
    if relocations_gdf is None or relocations_gdf.empty:
        raise ValueError("Relocations gdf is empty.")
    if "groupby_col" not in relocations_gdf.columns:
        raise ValueError("Relocations gdf must contain 'groupby_col' column.")
    if time_column not in relocations_gdf.columns:
        raise ValueError(f"Relocations gdf must contain '{time_column}' column.")

    df = relocations_gdf[["groupby_col", time_column]].copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.dropna(subset=[time_column])
    span = df.groupby("groupby_col", dropna=False)[time_column].agg(first="min", last="max").reset_index()
    span["mature"] = span["last"] >= (span["first"] + pd.DateOffset(months=months_duration))
    subject_df = subject_df.merge(span[["groupby_col", "mature"]], on="groupby_col", how="left")
    subject_df["mature"] = subject_df["mature"].fillna(False)
    return subject_df


@task
def compute_subject_stats(
    traj_gdf: AnyGeoDataFrame,
    subject_df: AnyDataFrame,
    etd_df: AnyGeoDataFrame,
    groupby_col: str,
    output_path: Union[str, Path] = None,
) -> AnyDataFrame:
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = remove_file_scheme(output_path)

    if traj_gdf is None or traj_gdf.empty:
        raise ValueError("`traj_gdf` is empty.")

    if "geometry" not in traj_gdf.columns:
        raise ValueError("`traj_gdf` must have a geometry column.")

    if groupby_col not in traj_gdf.columns:
        raise ValueError(f"`traj_gdf` must contain '{groupby_col}'.")

    if "segment_start" not in traj_gdf.columns:
        raise ValueError("`traj_gdf` must contain a 'segment_start' datetime column.")

    # Parse datetime column
    traj_gdf = traj_gdf.copy()
    traj_gdf = traj_gdf.to_crs(epsg=32737)
    traj_gdf["segment_start"] = pd.to_datetime(traj_gdf["segment_start"], errors="coerce")
    if traj_gdf["segment_start"].isna().all():
        raise ValueError("All 'segment_start' values are NaT after parsing.")

    # Ensure single subject
    non_null_ids = traj_gdf[groupby_col].dropna().unique()
    if len(non_null_ids) == 0:
        raise ValueError(f"No non-null values found in '{groupby_col}'.")
    if len(non_null_ids) > 1:
        raise ValueError(
            f"Multiple subjects present in '{groupby_col}' (found {len(non_null_ids)}). "
            "Provide a single-subject GDF."
        )
    subject_id = non_null_ids[0]

    # Get maturity status
    mature = False
    if subject_df is not None and not subject_df.empty:
        if groupby_col in subject_df.columns and "mature" in subject_df.columns:
            mrow = subject_df.loc[subject_df[groupby_col] == subject_id, "mature"]
            if not mrow.empty:
                mature = bool(mrow.iloc[0])

    # Calculate MCP (Minimum Convex Polygon)
    hull_geom = traj_gdf.geometry.unary_union.convex_hull
    if hull_geom.is_empty:
        raise ValueError("Convex hull is empty. Check trajectory geometries.")
    hull_area_m2 = hull_geom.area

    mcp_km2 = round(hull_area_m2 / 1_000_000.0, 1)

    etd_km2 = 0.0

    if etd_df is None:
        logger.info("ETD DataFrame is None")
    elif etd_df.empty:
        logger.info("ETD DataFrame is empty")
    else:
        logger.info("ETD DataFrame provided")
        logger.info(f" - Shape: {etd_df.shape}")
        logger.info(f" - Columns: {etd_df.columns.tolist()}")
        logger.info(f" - CRS: {etd_df.crs}")

        etd_km2 = round(float(etd_df[etd_df["percentile"] >= 99.9]["area_sqkm"].sum()), 1)

    logger.info(f"Final ETD value: {etd_km2} km²")

    # Calculate time tracked
    tmin = traj_gdf["segment_start"].min()
    tmax = traj_gdf["segment_start"].max()
    delta = tmax - tmin
    time_tracked_days = int(delta.days)
    time_tracked_years = round(time_tracked_days / 365.25, 1)

    # Calculate distance travelled
    distance_travelled_km = 0.0
    if "dist_meters" in traj_gdf.columns and not traj_gdf["dist_meters"].isna().all():
        distance_travelled_km = round(float(traj_gdf["dist_meters"].sum()) / 1000.0, 1)

    # Calculate max displacement from first point
    traj_gdf = traj_gdf.sort_values("segment_start")
    first_geom = traj_gdf.geometry.iat[0]
    max_displacement_km = round(float(traj_gdf.geometry.distance(first_geom).max()) / 1000.0, 1)

    # Calculate night/day ratio
    traj_ll = traj_gdf.to_crs(4326)
    summary_params = [SummaryParam(display_name="night_day_ratio", aggregator="night_day_ratio")]
    summarized = analysis.summarize_df(traj_ll, summary_params, groupby_cols=None)
    night_day_ratio = round(float(summarized["night_day_ratio"].iloc[0]), 2)

    # get subject name
    name = traj_gdf["subject_name"].unique()[0]

    # Compile statistics
    stats = {
        "subject_id": str(subject_id),
        "name": str(name),
        "mature": bool(mature),
        "MCP": float(mcp_km2),
        "ETD": float(etd_km2),
        "time_tracked_days": int(time_tracked_days),
        "time_tracked_years": float(time_tracked_years),
        "distance_travelled": float(distance_travelled_km),
        "max_displacement": float(max_displacement_km),
        "night_day_ratio": float(night_day_ratio),
    }
    return cast(AnyDataFrame, pd.DataFrame([stats]))

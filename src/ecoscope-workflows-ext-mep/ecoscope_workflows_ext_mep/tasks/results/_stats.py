from ecoscope.platform.tasks.analysis._summary import SummaryParam
from ecoscope.platform.tasks import analysis
from wt_registry import register
import pandas as pd
from typing import cast
from ecoscope.platform.annotations import AnyDataFrame,AnyGeoDataFrame

@register()
def compute_subject_stats(
    traj_gdf: AnyGeoDataFrame,
    subject_df: AnyDataFrame,
    etd_df: AnyGeoDataFrame,
) -> AnyDataFrame:
    traj_gdf = traj_gdf.to_crs(epsg=32737)
    traj_gdf["segment_start"] = pd.to_datetime(traj_gdf["segment_start"], errors="coerce")

    # Ensure single subject
    non_null_ids = traj_gdf["groupby_col"].dropna().unique()
    if len(non_null_ids) == 0:
        raise ValueError(f"No non-null values found in 'groupby_col'.")
    if len(non_null_ids) > 1:
        raise ValueError(
            f"Multiple subjects present in 'groupby_col' (found {len(non_null_ids)}). "
            "Provide a single-subject GDF."
        )
    subject_id = non_null_ids[0]

    # Get maturity status
    mature = False
    if subject_df is not None and not subject_df.empty:
        if "groupby_col" in subject_df.columns and "mature" in subject_df.columns:
            mrow = subject_df.loc[subject_df["groupby_col"] == subject_id, "mature"]
            if not mrow.empty:
                mature = bool(mrow.iloc[0])

    # Calculate MCP (Minimum Convex Polygon)
    hull_geom = traj_gdf.geometry.union_all().convex_hull
    hull_area_m2 = hull_geom.area

    mcp_km2 = round(hull_area_m2 / 1_000_000.0, 1)
    etd_km2 = 0.0

    if etd_df is None:
        print("ETD DataFrame is None")
    elif etd_df.empty:
        print("ETD DataFrame is empty")
    else:
        etd_km2 = round(float(etd_df[etd_df["percentile"] >= 99.9]["area_sqkm"].sum()), 1)

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

    # Compile statistics
    stats = {
        "subject_id": str(subject_id),
        "mature": bool(mature),
        "mcp": float(mcp_km2),
        "etd": float(etd_km2),
        "time_tracked_days": int(time_tracked_days),
        "time_tracked_years": float(time_tracked_years),
        "distance_travelled": float(distance_travelled_km),
        "max_displacement": float(max_displacement_km),
        "night_day_ratio": float(night_day_ratio),
    }
    return cast(AnyDataFrame, pd.DataFrame([stats]))

import json
import hashlib
import ecoscope
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from pydantic.json_schema import SkipJsonSchema
from pydantic import Field, BaseModel, ConfigDict
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerClient
from typing import Sequence, Tuple, Union, Annotated, cast, Optional, Dict, List, Literal
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AdvancedField, AnyDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.analysis import calculate_elliptical_time_density

class AutoScaleGridCellSize(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Auto-scale"})
    auto_scale_or_custom: Annotated[
        Literal["Auto-scale"],
        AdvancedField(
            default="Auto-scale",
            title=" ",
            description="Define the resolution of the raster grid (in meters per pixel).",
        ),
    ] = "Auto-scale"

class CustomGridCellSize(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Customize"})
    auto_scale_or_custom: Annotated[
        Literal["Customize"],
        AdvancedField(
            default="Customize",
            title=" ",
            description="Define the resolution of the raster grid (in meters per pixel).",
        ),
    ] = "Customize"
    grid_cell_size: Annotated[
        float | SkipJsonSchema[None],
        Field(
            description="Custom Raster Pixel Size (Meters)",
            gt=0,
            lt=10000,
            default=5000,
            json_schema_extra={"exclusiveMinimum": 0, "exclusiveMaximum": 10000},
        ),
    ] = 5000


@task
def get_area_bounds(
    gdf: AnyGeoDataFrame,
    names: Union[str, Sequence[str]],
    column: str,
) -> Tuple[float, float, float, float]:
    """
    Return (xmin, ymin, xmax, ymax) for all rows where `gdf[column]` matches any of `names`.

    Parameters
    ----------
    gdf : AnyGeoDataFrame
        GeoDataFrame containing polygon geometries.
    names : str or list of str
        Value(s) to match in the specified column.
    column : str
        Column name to filter on.

    Returns
    -------
    (xmin, ymin, xmax, ymax) : tuple of float
        Bounding box coordinates for the selected rows.
    """
    if column not in gdf.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(gdf.columns)}")

    # Normalize names to a list
    if isinstance(names, str):
        names = [names]
    if not names:
        raise ValueError("`names` must not be empty.")

    subset = gdf[gdf[column].isin(names)]
    if subset.empty:
        raise ValueError(f"No match found for {names} in column '{column}'.")

    xmin, ymin, xmax, ymax = subset.total_bounds
    return xmin, ymin, xmax, ymax


@task
def get_subjects_info(
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
        id=",".join(ids) if ids else None,
        max_ids_per_request=max_ids_per_request,
    )

    if raise_on_empty and df.empty:
        raise ValueError("No data returned from EarthRanger for get_subjects")

    return df


@task
def download_profile_photo(
    df: AnyDataFrame,
    column: str,
    output_path: Union[str, Path],
    image_type: str = ".png",
    overwrite_existing: bool = True,
) -> Optional[str]:
    """
    Download profile photos from URLs in a DataFrame column.

    Returns:
        Path to the last successfully downloaded file if successful, None otherwise
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, url in df[column].dropna().items():
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            print(f"Skipping invalid URL at index {idx}: {url}")
            continue

        try:
            df_subset = df.loc[[idx]]
            row_hash = hashlib.sha256(
                pd.util.hash_pandas_object(df_subset, index=True).values
            ).hexdigest()
            filename = f"{row_hash[:8]}_{idx}"

            if not image_type.startswith("."):
                image_type = f".{image_type}"

            file_path = output_path / f"{filename}{image_type}"
            processed_url = url.replace("dl=0", "dl=1") if "dropbox.com" in url else url
            ecoscope.io.utils.download_file(processed_url, str(file_path), overwrite_existing)
            print(f"Downloaded profile photo for index {idx} to {file_path}")
        except Exception as e:
            print(f"Error processing URL at index {idx} ({url}): {e}")
            continue

    return str(file_path) if 'file_path' in locals() else None

def safe_strip(x) -> str:
    return "" if x is None else str(x).strip()

def truncate_at_sentence(text: str, maxlen: int) -> str:
    """Truncate text at sentence boundary within maxlen."""
    if len(text) <= maxlen:
        return text

    last_period_index = text[:maxlen].rfind(". ")
    if last_period_index != -1:
        return text[: last_period_index + 1]

    for ending in ["! ", "? ", ": "]:
        last_ending = text[:maxlen].rfind(ending)
        if last_ending != -1:
            return text[: last_ending + 1]

    last_space = text[:maxlen].rfind(" ")
    if last_space != -1:
        return text[:last_space] + "..."

    return text[:maxlen] + "..."

def truncate_at_sentence(text: str, maxlen: int) -> str:
    if len(text) <= maxlen:
        return text
    cut = text[:maxlen]
    dot = cut.rfind(".")
    return (cut[: dot + 1] if dot >= 40 else cut.rstrip()) + ("..." if dot < 40 else "")

def format_date(date_str: str) -> str:
    s = safe_strip(date_str)
    if not s:
        return ""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%b %d, %Y", "%d %b %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%d %b %Y")
        except ValueError:
            continue
    return s

def save_as_json(data: Union[Dict, List], output_path: Path) -> None:
    """Save data as JSON file."""
    if not str(output_path).endswith(".json"):
        output_path = output_path.with_suffix(".json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@task
def persist_subject_info(
    df: AnyDataFrame,                    
    output_path: Union[str, Path],
    maxlen: int = 1000,
    return_data: bool = True,
) -> Optional[Union[Dict[str, str], List[Dict[str, str]]]]:
    if df.empty:
        raise ValueError("DataFrame is empty")

    required_columns = [
        "subject_name",
        "additional__Bio",
        "additional__DOB",
        "additional__sex",
        "additional__notes",
        "additional__country",
        "additional__distribution",
        "additional__status",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns {missing} don't exist in the dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)  # make the target dir itself

    def process_single_subject(row: pd.Series) -> Dict[str, str]:
        bio = safe_strip(row.get("additional__Bio", ""))
        if len(bio) > maxlen:
            bio = truncate_at_sentence(bio, maxlen)

        status_value = safe_strip(row.get("additional__status", ""))
        status_color = "green" if status_value.lower() == "active" else "red"

        dob_formatted = format_date(safe_strip(row.get("additional__DOB", "")))

        subject_info = {
            "subject_name": safe_strip(row.get("subject_name", "")).title(),
            "dob": dob_formatted,
            "sex": safe_strip(row.get("additional__sex", "")).capitalize(),
            "country": safe_strip(row.get("additional__country", "")),
            "notes": safe_strip(row.get("additional__notes", "")),
            "status": f'<span style="color: {status_color};">{status_value}</span>',
            "status_raw": status_value,
            "bio": bio,
            "distribution": safe_strip(row.get("additional__distribution", "")),
        }
        return {k: ("" if v is None else str(v)) for k, v in subject_info.items()}

    processed_data: Union[Dict[str, str], List[Dict[str, str]]]
    if len(df) == 1:
        processed_data = process_single_subject(df.iloc[0])
    else:
        processed_data = [process_single_subject(row) for _, row in df.iterrows()]

    # Persist JSON with a stable, collision-resistant filename
    try:
        # Hash the input frame (values+index) so the filename reflects the content
        df_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
        filename = f"subject_info_{df_hash[:8]}.json"
        file_path = output_path / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        print(f"Subject info successfully saved to: {file_path}")

    except Exception as e:
        print(f"Error saving subject info: {e}")
        raise

    return processed_data if return_data else None

@task
def split_gdf_by_column(
    gdf: Annotated[AnyGeoDataFrame, Field(description="The GeoDataFrame to split")],
    column: Annotated[str, Field(description="Column name to split GeoDataFrame by")],
) -> Dict[str, AnyGeoDataFrame]:
    """
    Splits a GeoDataFrame into a dictionary of GeoDataFrames based on unique values in the specified column.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to split.
        column (str): The column to split by.

    Returns:
        Dict[str, gpd.GeoDataFrame]: Dictionary where keys are unique values of the column are GeoDataFrames.
    """
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found in GeoDataFrame.")

    grouped = {str(k): v for k, v in gdf.groupby(column)}
    return grouped


@task
def generate_mcp_gdf(
    gdf: AnyGeoDataFrame,
    planar_crs: str = "ESRI:102022",  # Africa Albers Equal Area
) -> AnyGeoDataFrame:
    """
    Create a Minimum Convex Polygon (MCP) from input point geometries and compute its area.
    """
    if gdf is None or gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")
    if gdf.geometry is None:
        raise ValueError("Input GeoDataFrame has no 'geometry' column.")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS set (e.g., EPSG:4326).")

    original_crs = gdf.crs

    # Filter out empty or null geometries
    valid_points_gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    if valid_points_gdf.empty:
        raise ValueError("No valid geometries in input GeoDataFrame.")

    # Ensure only points (fallback: centroid of non-point geometries)
    if not all(valid_points_gdf.geometry.geom_type.isin(["Point"])):
        valid_points_gdf.geometry = valid_points_gdf.geometry.centroid

    projected_gdf = valid_points_gdf.to_crs(planar_crs)
    convex_hull = projected_gdf.geometry.unary_union.convex_hull

    area_sq_meters = float(convex_hull.area)
    area_sq_km = area_sq_meters / 1_000_000.0
    convex_hull_original_crs = gpd.GeoSeries([convex_hull], crs=planar_crs).to_crs(original_crs).iloc[0]

    result_gdf = gpd.GeoDataFrame(
        {"area_m2": [area_sq_meters], "area_km2": [area_sq_km], "mcp": "mcp"},
        geometry=[convex_hull_original_crs],
        crs=original_crs,
    )
    return result_gdf


@task
def calculate_etd_by_groups(
    trajectory_gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="The trajectory geodataframe.", exclude=True),
    ],
    groupby_cols: Annotated[
        list[str],
        Field(
            description="List of column names to group by (e.g., ['groupby_col', 'extra__name'])",
            json_schema_extra={"default": ["groupby_col", "extra__name"]},
        ),
    ] = None,
    auto_scale_or_custom_cell_size: Annotated[
        AutoScaleGridCellSize | CustomGridCellSize | SkipJsonSchema[None],
        Field(
            json_schema_extra={
                "title": "Auto Scale Or Custom Grid Cell Size",
                "ecoscope:advanced": True,
                "default": {"auto_scale_or_custom": "Auto-scale"},
            },
        ),
    ] = None,
    crs: Annotated[
        str,
        AdvancedField(
            default="EPSG:3857",
            title="Coordinate Reference System",
            description="The coordinate reference system in which to perform the density calculation",
        ),
    ] = "EPSG:3857",
    nodata_value: Annotated[float | str, AdvancedField(default="nan")] = "nan",
    band_count: Annotated[int, AdvancedField(default=1)] = 1,
    max_speed_factor: Annotated[
        float,
        AdvancedField(
            default=1.05,
            title="Max Speed Factor (Kilometers per Hour)",
            description="An estimate of the subject's maximum speed.",
        ),
    ] = 1.05,
    expansion_factor: Annotated[
        float,
        AdvancedField(
            default=1.3,
            title="Shape Buffer Expansion Factor",
            description="Controls how far time density values spread across the grid.",
        ),
    ] = 1.3,
    percentiles: Annotated[
        list[float] | SkipJsonSchema[None],
        Field(default=[25.0, 50.0, 75.0, 90.0, 95.0, 99.9]),
    ] = None,
    include_groups: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to include grouping columns in the result",
        ),
    ] = False,
) -> AnyDataFrame:
    """
    Calculate Elliptical Time Density (ETD) for trajectory groups.

    This function applies calculate_elliptical_time_density to each group
    defined by the groupby_cols, similar to:

    trajs.groupby(["groupby_col", "extra__name"]).apply(
        lambda df: calculate_elliptical_time_density(df, ...),
        include_groups=False,
    )

    Args:
        trajectory_gdf: The trajectory geodataframe
        groupby_cols: List of column names to group by
        **kwargs: All other parameters passed to calculate_elliptical_time_density

    Returns:
        DataFrame with ETD results for all groups combined
    """

    # Set default groupby columns if not provided
    if groupby_cols is None:
        groupby_cols = ["groupby_col", "extra__name"]

    # Set default percentiles if not provided
    if percentiles is None:
        percentiles = [25.0, 50.0, 75.0, 90.0, 95.0, 99.9]

    # Validate that groupby columns exist
    missing_cols = [col for col in groupby_cols if col not in trajectory_gdf.columns]
    if missing_cols:
        raise ValueError(f"Groupby columns {missing_cols} not found in trajectory_gdf")

    def apply_etd_to_group(group_df):
        """Apply calculate_elliptical_time_density to a single group"""
        try:
            result = calculate_elliptical_time_density(
                trajectory_gdf=group_df,
                auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
                crs=crs,
                nodata_value=nodata_value,
                band_count=band_count,
                max_speed_factor=max_speed_factor,
                expansion_factor=expansion_factor,
                percentiles=percentiles,
            )
            return result
        except Exception as e:
            print(f"Failed to calculate ETD : {group_df.name if hasattr(group_df, 'name') else 'unknown'}: {e}")
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                {
                    "percentile": pd.Series(dtype="float64"),
                    "geometry": gpd.GeoSeries(dtype="geometry"),
                    "area_sqkm": pd.Series(dtype="float64"),
                }
            )

    # Apply ETD calculation to each group
    try:
        grouped_results = trajectory_gdf.groupby(groupby_cols).apply(apply_etd_to_group, include_groups=include_groups)

        # Reset index to get a clean DataFrame
        if include_groups:
            result = grouped_results.reset_index()
        else:
            result = grouped_results.reset_index(level=groupby_cols, drop=True).reset_index(drop=True)

        return cast(AnyDataFrame, result)

    except Exception as e:
        print(f"Failed to calculate ETD by groups: {e}")
        empty_result = pd.DataFrame(
            {
                "percentile": pd.Series(dtype="float64"),
                "geometry": gpd.GeoSeries(dtype="geometry"),
                "area_sqkm": pd.Series(dtype="float64"),
            }
        )
        return cast(AnyDataFrame, empty_result)


@task
def create_seasonal_labels(traj: AnyGeoDataFrame, total_percentiles: AnyDataFrame) -> Optional[AnyGeoDataFrame]:
    """
    Annotates trajectory segments with seasonal labels (wet/dry) based on NDVI-derived windows.
    Applies to the entire trajectory without grouping.
    """
    try:
        print("Calculating seasonal ETD percentiles for entire trajectory")
        print(f"Total percentiles shape: {total_percentiles.shape}")
        print(f"Available seasons: {total_percentiles['season'].unique()}")

        # Since total_percentiles contains the seasonal windows directly,
        # we don't need determine_season_windows() - we can use it directly
        seasonal_wins = total_percentiles.copy()

        # Filter to trajectory time range if needed
        traj_start = traj["segment_start"].min()
        traj_end = traj["segment_end"].max()

        # Keep only seasonal windows that overlap with trajectory timeframe
        seasonal_wins = seasonal_wins[
            (seasonal_wins["end"] >= traj_start) & (seasonal_wins["start"] <= traj_end)
        ].reset_index(drop=True)

        print(f"Filtered seasonal windows: {len(seasonal_wins)} periods")
        print(f"Seasonal Windows:\n{seasonal_wins[['start', 'end', 'season']]}")

        if seasonal_wins.empty:
            print("No seasonal windows overlap with trajectory timeframe.")
            traj["season"] = None
            return traj

        # Create interval index
        season_bins = pd.IntervalIndex(data=seasonal_wins.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1))
        print(f"Created {len(season_bins)} seasonal bins")

        labels = seasonal_wins["season"].values

        # Use pd.cut to assign segments to seasonal bins
        traj["season"] = pd.cut(traj["segment_start"], bins=season_bins, include_lowest=True).map(
            dict(zip(season_bins, labels))
        )

        # Handle segments that fall outside seasonal windows
        null_count = traj["season"].isnull().sum()
        if null_count > 0:
            print(f"Warning: {null_count} trajectory segments couldn't be assigned to any season")

        print("Seasonal labeling complete. Season distribution:")
        print(traj["season"].value_counts(dropna=False))

        return traj

    except Exception as e:
        print(f"Failed to apply seasonal label to trajectory: {e}")
        import traceback

        traceback.print_exc()
        return None

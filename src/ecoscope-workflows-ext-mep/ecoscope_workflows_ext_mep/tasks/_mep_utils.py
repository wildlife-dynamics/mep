import os
import json
import jinja2
import hashlib
import ecoscope
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PyPDF2 import PdfMerger
from datetime import datetime
import plotly.graph_objects as go
from urllib.parse import urlparse
from plotly.graph_objs import Figure
from ecoscope.io import download_file
from ecoscope.plotting.plot import nsd
from urllib.request import url2pathname
from plotly.subplots import make_subplots
from ecoscope.trajectory import Trajectory
from ecoscope.relocations import Relocations
from pydantic.json_schema import SkipJsonSchema
from pydantic import Field, BaseModel, ConfigDict
from shapely.geometry import Polygon, MultiPolygon
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.tasks import analysis
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerClient
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import ExportArgs
from ecoscope_workflows_ext_ecoscope.tasks.analysis._summary import SummaryParam
from typing import Sequence, Tuple, Union, Annotated, cast, Optional, Dict, List, Literal,Any
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
    if output_path.startswith("file://"):
        parsed = urlparse(output_path)
        output_path = url2pathname(parsed.path)
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
            row_hash = hashlib.sha256(pd.util.hash_pandas_object(df_subset, index=True).values).hexdigest()
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

    return str(file_path) if "file_path" in locals() else None


def safe_strip(x) -> str:
    return "" if x is None else str(x).strip()

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

@task
def persist_subject_info(
    df: AnyDataFrame,
    output_path: Union[str, Path],
    maxlen: int = 1000,
    return_data: bool = True,
) -> Optional[Union[Dict[str, str], List[Dict[str, str]]]]:
    if df.empty:
        raise ValueError("DataFrame is empty")

    if output_path.startswith("file://"):
        parsed = urlparse(output_path)
        output_path = url2pathname(parsed.path)

    required_columns = [
        "subject_name",
        "subject_bio",
        "date_of_birth",
        "subject_sex",
        "notes",
        "country",
        "distribution",
        "status",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns {missing} don't exist in the dataframe. " f"Available columns: {list(df.columns)}"
        )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)  # make the target dir itself

    def process_single_subject(row: pd.Series) -> Dict[str, str]:
        bio = safe_strip(row.get("subject_bio", ""))
        if len(bio) > maxlen:
            bio = truncate_at_sentence(bio, maxlen)

        status_value = safe_strip(row.get("status", ""))
        status_color = "green" if status_value.lower() == "active" else "red"

        dob_formatted = format_date(safe_strip(row.get("date_of_birth", "")))

        subject_info = {
            "subject_name": safe_strip(row.get("subject_name", "")).title(),
            "dob": dob_formatted,
            "sex": safe_strip(row.get("subject_sex", "")).capitalize(),
            "country": safe_strip(row.get("country", "")),
            "notes": safe_strip(row.get("notes", "")),
            "status": f'<span style="color: {status_color};">{status_value}</span>',
            "status_raw": status_value,
            "bio": bio,
            "distribution": safe_strip(row.get("distribution", "")),
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


def add_seasons_square(fig: Figure, dataframe: AnyDataFrame) -> Figure:
    """
    Add shaded seasonal rectangles and labels to a Plotly time series figure.

    Args:
        fig (Figure): A Plotly figure object (typically from NSD plotting).
        season_df (AnyDataFrame): DataFrame with columns ['start', 'end', 'season'].

    Returns:
        Figure: The updated Plotly figure with season annotations.
    """

    # Convert to subplot if not already
    if not hasattr(fig, "_grid_ref"):  # Avoid re-wrapping
        fig = make_subplots(figure=fig, specs=[[{"secondary_y": True}]])

    for _, row in dataframe.iterrows():
        try:
            start_dt = pd.to_datetime(row["start"])
            end_dt = pd.to_datetime(row["end"])
            if end_dt <= start_dt:
                continue  # Skip invalid intervals
        except Exception:
            continue  # Skip malformed datetime rows

        season_type = row.get("season", "").strip().lower()

        # Season-based fill color
        fillcolor = {"wet": "rgba(0,0,255,0.25)", "dry": "rgba(0,0,255,0.05)"}.get(season_type, "rgba(0,0,0,0.05)")

        fig.add_shape(
            type="rect",
            x0=start_dt,
            x1=end_dt,
            y0=0,
            y1=1,
            yref="paper",
            fillcolor=fillcolor,
            line_width=0,
            layer="below",
        )

        midpoint = start_dt + (end_dt - start_dt) / 2
        fig.add_annotation(
            x=midpoint,
            y=1.02,
            text=row.get("season", "Season").capitalize(),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=12),
        )

    return fig

def _load_seasons_df(seasons_df: Union[str, Path, AnyDataFrame]) -> AnyDataFrame:
    if seasons_df is None:
        raise ValueError("Seasonal windows input is None.")
    p = Path(seasons_df)
    if not p.exists():
        raise FileNotFoundError(f"Seasonal windows file not found: {p}")
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    elif p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    else:
        return pd.read_csv(p)

def _validate_seasons_df(df: AnyDataFrame) -> None:
    if df.empty:
        raise ValueError("Seasonal windows DataFrame is empty.")

@task
def generate_seasonal_nsd_plot(
    gdf: AnyGeoDataFrame,
    seasons_df: Union[str, Path, AnyDataFrame],
    widget_id: Annotated[
        str | None,
        Field(
            description=(
                "The id of the dashboard widget that this tile layer belongs to. "
                "If set this MUST match the widget title as defined downstream in create_widget tasks"
            ),
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    if gdf is None or getattr(gdf, "empty", True):
        raise ValueError("Input GeoDataFrame is empty.")

    seasons_df = _load_seasons_df(seasons_df)
    _validate_seasons_df(seasons_df)

    gdf = Relocations.from_gdf(gdf)
    figure = nsd(gdf)
    figure = add_seasons_square(figure, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@task
def generate_seasonal_speed_plot(
    gdf: AnyGeoDataFrame,
    seasons_df: Union[str, Path, AnyDataFrame],
    widget_id: Annotated[
        str | None,
        Field(
            description=(
                "The id of the dashboard widget that this tile layer belongs to. "
                "If set this MUST match the widget title as defined downstream in create_widget tasks"
            ),
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    if gdf is None or getattr(gdf, "empty", True):
        raise ValueError("Input GeoDataFrame is empty.")

    seasons_df = _load_seasons_df(seasons_df)
    _validate_seasons_df(seasons_df)

    gdf = Relocations.from_gdf(gdf)
    trajs = Trajectory.from_relocations(gdf)
    figure = ecoscope.plotting.speed(trajs)
    figure = add_seasons_square(figure, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@task
def generate_seasonal_mcp_asymptote_plot(
    gdf: AnyGeoDataFrame,
    seasons_df: Union[str, Path, AnyDataFrame],
    widget_id: Annotated[
        str | None,
        Field(
            description=(
                "The id of the dashboard widget that this tile layer belongs to. "
                "If set this MUST match the widget title as defined downstream in create_widget tasks"
            ),
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    if gdf is None or getattr(gdf, "empty", True):
        raise ValueError("Input GeoDataFrame is empty.")

    seasons_df = _load_seasons_df(seasons_df)
    _validate_seasons_df(seasons_df)

    gdf = Relocations.from_gdf(gdf)
    figure = ecoscope.plotting.mcp(gdf)
    figure = add_seasons_square(figure, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

def collar_event_timeline_plot(
    geodataframe: AnyGeoDataFrame,
    collar_events: Optional[AnyDataFrame] = None,
) -> go.FigureWidget:
    """
    Generates an interactive timeline plot of collar events overlaid on subject relocations.

    Args:
        geodataframe (GeoDataFrame): Subject relocations with 'fixtime' datetime column.
        collar_events (pd.DataFrame, optional): Collar event log with 'time', 'event_type', 
                                                and 'priority_label' columns.

    Returns:
        go.FigureWidget: A Plotly timeline figure widget.
    """
    fig = go.FigureWidget()
    
    # Validate and clean relocations
    geodataframe = geodataframe.dropna(subset=["fixtime"]).copy()
    if geodataframe.empty:
        raise ValueError("No valid relocations after removing NaT values.")
    
    # Determine time range
    time_min = geodataframe["fixtime"].min()
    time_max = geodataframe["fixtime"].max()
    
    max_event_y = 0
    
    # Plot events if available
    if collar_events is not None and not collar_events.empty:
        collar_events = collar_events.dropna(subset=["time"]).copy()
        
        if not collar_events.empty:
            # Plot event lines and markers
            for i, (_, row) in enumerate(collar_events.iterrows(), 1):
                fig.add_trace(go.Scatter(
                    x=[row["time"], row["time"], row["time"]],
                    y=[0, i, 0],
                    mode="lines",
                    line=dict(color=row["priority_label"], width=2),
                    hovertemplate=(
                        f"<b>{row['event_type']}</b><br>"
                        f"{row['time'].strftime('%Y-%m-%d %H:%M:%S')}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    name=row["event_type"],
                ))
            
            # Plot event markers for better visibility
            fig.add_trace(go.Scatter(
                x=collar_events["time"],
                y=range(1, len(collar_events) + 1),
                mode="markers",
                marker=dict(
                    size=8,
                    color=collar_events["priority_label"],
                    symbol="circle",
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<extra></extra>"
                ),
                customdata=collar_events[["event_type", "time"]].values,
                showlegend=False,
            ))
            
            max_event_y = len(collar_events)
            time_min = min(time_min, collar_events["time"].min())
            time_max = max(time_max, collar_events["time"].max())
    
    # Plot relocations as baseline markers
    relocation_y = max_event_y + 0.5
    fig.add_trace(go.Scatter(
        x=geodataframe["fixtime"],
        y=np.full(len(geodataframe), relocation_y),
        mode="markers",
        marker=dict(size=4, color="rgb(0, 100, 200)"),
        hovertemplate="Relocation: %{x}<extra></extra>",
        showlegend=False,
    ))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=15),
        xaxis=dict(
            range=[time_min, time_max],
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGray",
        ),
        yaxis=dict(visible=False),
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
    )
    
    return fig

@task 
def generate_collared_seasonal_plot(
    events_gdf: AnyDataFrame,
    relocations_gdf: AnyGeoDataFrame,
    seasons_df: Union[str, Path, AnyDataFrame],
    filter_col: str,
    widget_id: Annotated[
        str | None,
        Field(
            description=(
                "The id of the dashboard widget that this tile layer belongs to. "
                "If set this MUST match the widget title as defined downstream in create_widget tasks"
            ),
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    seasons_df = _load_seasons_df(seasons_df)
    _validate_seasons_df(seasons_df)

    if relocations_gdf is None or relocations_gdf.empty:
        raise ValueError("Relocations GeoDataFrame is empty.")

    subject_name = relocations_gdf['subject_name'].unique()[0]

    if events_gdf is None or events_gdf.empty:
        print(f"No events data for subject '{subject_name}'. Generating plot with relocations only.")
        events_gdf = None
    elif filter_col not in events_gdf.columns:
        raise ValueError(f"Column '{filter_col}' not found. Available: {', '.join(events_gdf.columns)}")
    else:
        events_gdf = events_gdf[events_gdf[filter_col] == subject_name]
        if events_gdf.empty:
            print(f"No events found for subject '{subject_name}'. Generating plot with relocations only.")
            events_gdf = None
    
    # Generate visualization with or without events
    fig = collar_event_timeline_plot(relocations_gdf, events_gdf)
    figure = add_seasons_square(fig, seasons_df)
    
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

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

    if relocations_gdf is None or getattr(relocations_gdf, "empty", True):
        raise ValueError("Input GeoDataFrame is empty.")
    if "groupby_col" not in relocations_gdf.columns:
        raise ValueError("Input GeoDataFrame must contain 'groupby_col' column.")
    if time_column not in relocations_gdf.columns:
        raise ValueError(f"Input GeoDataFrame must contain '{time_column}' column.")

    df = relocations_gdf[["groupby_col", time_column]].copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.dropna(subset=[time_column])

    span = (
        df.groupby("groupby_col", dropna=False)[time_column]
          .agg(first="min", last="max")
          .reset_index()
    )
    span["mature"] = span["last"] >= (span["first"] + pd.DateOffset(months=months_duration))
    subject_df = subject_df.merge(span[["groupby_col", "mature"]], on="groupby_col", how="left")
    subject_df["mature"] = subject_df["mature"].fillna(False)
    return subject_df

def safe_json_serialize(obj: Any) -> Any:
    """Convert objects to JSON-serializable types"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj

# Fixed version of get_subject_stats function
@task
def get_subject_stats(
    traj_gdf: AnyGeoDataFrame,
    subject_df: AnyDataFrame,
    etd_df: AnyGeoDataFrame,
    output_path: Union[str, Path],
    groupby_col: str,
) -> Dict[str, Union[str, int, float, bool]]:
    import json, hashlib
    from pathlib import Path
    import numpy as np
    import pandas as pd

    if output_path.startswith("file://"):
        parsed = urlparse(output_path)
        output_path = url2pathname(parsed.path)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if traj_gdf is None or getattr(traj_gdf, "empty", True):
        raise ValueError("`traj_gdf` is empty.")

    if "geometry" not in traj_gdf.columns:
        raise ValueError("`traj_gdf` must have a geometry column.")

    if groupby_col not in traj_gdf.columns:
        raise ValueError(f"`traj_gdf` must contain '{groupby_col}'.")
        
    if "segment_start" not in traj_gdf.columns:
        raise ValueError("`traj_gdf` must contain a 'segment_start' datetime column.")

    traj_gdf = traj_gdf.copy()
    traj_gdf["segment_start"] = pd.to_datetime(traj_gdf["segment_start"], errors="coerce")
    if traj_gdf["segment_start"].isna().all():
        raise ValueError("All 'segment_start' values are NaT after parsing.")

    non_null_ids = traj_gdf[groupby_col].dropna().unique()
    if len(non_null_ids) == 0:
        raise ValueError(f"No non-null values found in '{groupby_col}'.")
    if len(non_null_ids) > 1:
        raise ValueError(
            f"Multiple subjects present in '{groupby_col}' (found {len(non_null_ids)}). "
            "Provide a single-subject GDF."
        )
    subject_id = non_null_ids[0]

    mature = False
    if subject_df is not None and subject_df.empty:
        if groupby_col in subject_df.columns and "mature" in subject_df.columns:
            mrow = subject_df.loc[subject_df[groupby_col] == subject_id, "mature"]
            if not mrow.empty:
                mature = bool(mrow.iloc[0])

    hull_geom = traj_gdf.geometry.unary_union.convex_hull
    if hull_geom.is_empty:
        raise ValueError("Convex hull is empty. Check trajectory geometries.")
    hull_area_m2 = hull_geom.area
    mcp_km2 = round(hull_area_m2 / 1_000_000.0, 1)

    etd_km2 = 0.0
    if etd_df is not None and etd_df.empty:
        if groupby_col in etd_df.columns:
            etd_sub = etd_df.loc[etd_df[groupby_col] == subject_id]
            if not etd_sub.empty:
                if "area" in etd_sub.columns:
                    etd_km2 = round(float(etd_sub["area"].sum()), 1)

    tmin = traj_gdf["segment_start"].min()
    tmax = traj_gdf["segment_start"].max()
    delta = tmax - tmin
    time_tracked_days = int(delta.days)
    time_tracked_years = round(time_tracked_days / 365.25, 1)

    distance_travelled_km = 0.0  
    if "dist_meters" in traj_gdf.columns and not traj_gdf["dist_meters"].isna().all():
        distance_travelled_km = round(float(traj_gdf["dist_meters"].sum()) / 1000.0, 1)

    traj_gdf = traj_gdf.sort_values("segment_start")
    first_geom = traj_gdf.geometry.iloc[0]
    max_displacement_km = round(
        float(traj_gdf.geometry.distance(first_geom).max()) / 1000.0, 1
    )

    traj_ll = traj_gdf.to_crs(4326)
    summary_params = [SummaryParam(display_name="night_day_ratio", aggregator="night_day_ratio")]
    summarized = analysis.summarize_df(traj_ll, summary_params, groupby_cols=None)
    night_day_ratio = round(float(summarized["night_day_ratio"].iloc[0]), 2)

    stats = {
        "subject_id": str(subject_id), 
        "mature": bool(mature),   
        "MCP": float(mcp_km2), 
        "ETD": float(etd_km2), 
        "time_tracked_days": int(time_tracked_days),
        "time_tracked_years": float(time_tracked_years), 
        "distance_travelled": float(distance_travelled_km),
        "max_displacement": float(max_displacement_km),
        "night_day_ratio": float(night_day_ratio), 
    }

    try:
        hash_input = f"{subject_id}_{tmin.isoformat()}_{tmax.isoformat()}".encode('utf-8')
        df_hash = hashlib.sha256(hash_input).hexdigest()
        out_path = output_dir / f"subject_stats_{df_hash[:8]}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"Subject stats saved to: {out_path}")
    except Exception as e:
        print(f"Warning: failed to write JSON ({e}).")
    return stats
    
@task
def calculate_seasonal_home_range(
    gdf: AnyGeoDataFrame,
    groupby_cols: Annotated[
        list[str],
        Field(
            description="List of column names to group by (e.g., ['groupby_col', 'season'])",
            json_schema_extra={"default": ["groupby_col", "season"]},
        ),
    ] = None,
    percentiles: Annotated[
        list[float] | SkipJsonSchema[None],
        Field(default=[25.0, 50.0, 75.0, 90.0, 95.0, 99.9]),
    ] = [99.9],
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
) -> AnyDataFrame:

    if groupby_cols is None:
        groupby_cols = ["groupby_col", "season"]
    
    if gdf is None or gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")
    
    if 'season' not in gdf.columns:
        raise ValueError("Input GeoDataFrame must have a 'season' column.")
    
    if auto_scale_or_custom_cell_size is None:
        auto_scale_or_custom_cell_size = AutoScaleGridCellSize()

    gdf = gdf[gdf['season'].notna()].copy()
    group_counts = gdf.groupby(groupby_cols).size()
    try:
        season_etd = gdf.groupby(groupby_cols).apply(
            lambda df: calculate_elliptical_time_density(
                df, 
                auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
                percentiles=percentiles,
            )
        )
    except TypeError:
        season_etd = gdf.groupby(groupby_cols).apply(
            lambda df: calculate_elliptical_time_density(
                df, 
                auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
                percentiles=percentiles,
            ),
            include_groups=False,
        )
    # Reset index properly
    if isinstance(season_etd.index, pd.MultiIndex):
        season_etd = season_etd.reset_index()

    return season_etd
    
# very specific to LandDx db 
@task
def build_template_region_lookup(
    gdf: AnyGeoDataFrame,
    categories: Optional[Dict[str, List[str]]] = None,
    static_ids: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Build a lookup dictionary grouping region IDs by category type.
    """
    if gdf is None or gdf.empty:
        raise ValueError("gdf cannot be empty.")

    gdf = gdf.reset_index()

    # Default categories (you can add more later)
    categories = categories or {
        "national_pa_use": [
            "National Park", "National Reserve",
            "National Reserve_Privately Managed",
            "National Reserve_Beacon_Adjusted", "Forest Reserve"
        ],
        "community_pa_use": ["Community Conservancy", "Group Ranch"],
    }

    # Default static UUIDs
    static_ids = static_ids or {
        "crop_raid_percent": ["2d3f6392-700c-495f-8bc5-087538f6f125"],
        "kenya_use": ["7895ded1-df29-4ca1-8e34-ebc8e3cbb24e"],
    }
    result = {}
    for name, types in categories.items():
        mask = gdf["type"].isin(types) & gdf["type"].notna() & ~gdf.is_empty
        result[name] = gdf.loc[mask, "globalid"].dropna().tolist()

    # Merge static IDs
    result.update(static_ids)

    return result

@task
def compute_template_regions(
    geodataframe: AnyGeoDataFrame, 
    template_lookup: Dict[str, list[str]], 
    crs: str
) -> Dict[str, Polygon | MultiPolygon]:
    return {
        template: geodataframe.query("globalid in @gids").to_crs(crs).unary_union
        for template, gids in template_lookup.items()
    }

@task
def compute_subject_occupancy(
    subjects_df: AnyDataFrame,
    crs: str,
    etd_gdf: AnyGeoDataFrame,
    regions_gdf: Dict[str, Polygon | MultiPolygon],
    output_path: Union[str, Path],
) -> Dict[str, float]:
    if subjects_df is None or subjects_df.empty:
        raise ValueError("Subjects dataframe is empty.")
    if etd_gdf is None or etd_gdf.empty:
        raise ValueError("ETD GeoDataFrame is empty.")
    if regions_gdf is None or not regions_gdf:
        raise ValueError("Regions dictionary is empty.")

    if isinstance(output_path, str) and output_path.startswith("file://"):
        parsed = urlparse(output_path)
        output_path = url2pathname(parsed.path)
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subject_id = subjects_df['subject_name'].iloc[0]
    
    # Get home range at 99.9th percentile and convert to target CRS
    try:
        subject_range = etd_gdf[etd_gdf['percentile'] == 99.9].to_crs(crs).geometry.iloc[0]
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"Could not extract 99.9th percentile ETD for subject '{subject_id}'. "
            f"Available percentiles: {etd_gdf['percentile'].unique().tolist()}"
        ) from e
    
    if subject_range.is_empty:
        raise ValueError(f"Home range geometry is empty for subject '{subject_id}'.")
    
    occupancy = {
        k: 100 * region.intersection(subject_range).area / subject_range.area
        for k, region in regions_gdf.items()
    }

    occupancy["unprotected"] = 100.0 - occupancy.get("national_pa_use", 0) - occupancy.get("community_pa_use", 0)
    occupancy = {k: round(v, 1) for k, v in occupancy.items()}
    
    # Save to file
    try:
        df_hash = hashlib.sha256(
            pd.util.hash_pandas_object(subjects_df, index=True).values
        ).hexdigest()
        filename = f"occupancy_{df_hash[:8]}.json"
        file_path = output_path / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(occupancy, f, indent=2, ensure_ascii=False)
        
        print(f"Subject occupancy for '{subject_id}' saved to: {file_path}")
        
    except Exception as e:
        raise Exception(f"Failed to save occupancy data for '{subject_id}': {e}") from e
    
    return occupancy


@task
def download_file_and_persist(
    url: Annotated[str, Field(description="URL to download the file from")],
    output_path: Annotated[str, Field(description="Path to save the downloaded file or directory")],
    retries: Annotated[int, Field(description="Number of retries on failure", ge=0)] = 3,
    overwrite_existing: Annotated[bool, Field(description="Whether to overwrite existing files")] = False,
    unzip: Annotated[bool, Field(description="Whether to unzip the file if it's a zip archive")] = False,
) -> str:
    """
    Downloads a file from the provided URL and persists it locally.
    Returns the full path to the downloaded (and optionally unzipped) file.
    """
    if output_path.startswith("file://"):
        parsed = urlparse(output_path)
        output_path = url2pathname(parsed.path)

    output_dir = os.path.isdir(output_path)

    # Determine expected filename BEFORE download
    if output_dir:
        import requests, email
        try:
            s = requests.Session()
            r = s.head(url, allow_redirects=True, timeout=10)
            m = email.message.Message()
            m["content-type"] = r.headers.get("content-disposition", "")
            filename = m.get_param("filename")
            if filename is None:
                filename = os.path.basename(urlparse(url).path.split("?")[0])
            if not filename:  # Fallback if still empty
                filename = "downloaded_file"
        except Exception:
            # If HEAD request fails, extract from URL
            filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"
        
        # Pass the full path to download_file, not just the directory
        target_path = os.path.join(output_path, filename)
    else:
        target_path = output_path

    # Perform download with the specific file path
    download_file(
        url=url,
        path=target_path,
        retries=retries,
        overwrite_existing=overwrite_existing,
        unzip=unzip
    )

    # If unzipped, find the extracted files
    if unzip and os.path.isdir(target_path.replace('.zip', '')):
        persisted_path = str(Path(target_path.replace('.zip', '')).resolve())
    else:
        persisted_path = str(Path(target_path).resolve())

    if not os.path.exists(persisted_path):
        # List what's actually in the directory for debugging
        parent_dir = os.path.dirname(persisted_path)
        if os.path.exists(parent_dir):
            actual_files = os.listdir(parent_dir)
            raise FileNotFoundError(
                f"Download failed — {persisted_path} not found after execution. "
                f"Files in {parent_dir}: {actual_files}"
            )
        else:
            raise FileNotFoundError(f"Download failed — {persisted_path} not found after execution.")

    return persisted_path

@task
def initialize_jinja_env(input_path: str) -> jinja2.Environment:
    """
    Initialize and return a Jinja2 environment for template rendering.
    
    Args:
        input_path: Path to search for templates
        
    Returns:
        jinja2.Environment: Configured Jinja2 environment instance
    """
    template_loader = jinja2.FileSystemLoader(searchpath=input_path)
    template_env = jinja2.Environment(loader=template_loader)
    return template_env


from pydantic import BaseModel, Field, FilePath
from typing import List
import jinja2
import pandas as pd

class EcomapPaths(BaseModel):
    movement_ecomap: str = Field(..., description="Path to movement ecomap")
    range_ecomap: str = Field(..., description="Path to range ecomap")
    overview_map: str = Field(..., description="Path to overview map")

class PlotPaths(BaseModel):
    voltage_plot: str
    nsd_plot: str
    speed_plot: str
    mcp_plot: str
    collar_event_timeline_plot: str

class AssetPaths(BaseModel):
    logo: str
    subject_photo: str

class IndividualData(BaseModel):
    subject_info: pd.DataFrame
    subject_stats: pd.DataFrame
    occupancy_info: pd.DataFrame
    
    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrames

def report_context(
    subjects_df: pd.DataFrame,
    ecomap_paths: EcomapPaths,
    plot_paths: PlotPaths,
    asset_paths: AssetPaths,
    individual_data: IndividualData,
    templates: List[str],
    template_env: jinja2.Environment
):
    """Generate report context with validated inputs."""
    # Pydantic validates types automatically!
    pass
    
def report_context(
    subjects_df
    individual_subject_info - path
    individual_subject_stats - path
    logo - path
    subject_photo - path
    movement_ecomap -path
    range_ecomap - path
    overview_map -path
    voltage_plot -path
    nsd_plot -path
    speed_plot - path
    mcp_plot -path 
    collar_event_timeline_plot -path
    individual_occupancy_info
    templates: list
    template_env
):
    print("Generating context ")
    
    if subjects_df is None or subjects_df.empty:
        raise ValueError("subjects df is empty")
    



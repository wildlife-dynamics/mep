import os
import json
import time
import base64
import jinja2
import zipfile
import hashlib
import logging
import ecoscope
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PyPDF2 import PdfMerger
from datetime import datetime
from selenium import webdriver
from urllib.parse import urlparse
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from ecoscope.io import download_file
from ecoscope.plotting.plot import nsd
from urllib.request import url2pathname
from dataclasses import asdict,dataclass
from plotly.subplots import make_subplots
from ecoscope.trajectory import Trajectory
from ecoscope.relocations import Relocations
from pydantic.json_schema import SkipJsonSchema
from pydantic import Field, BaseModel, ConfigDict
from shapely.geometry import Polygon, MultiPolygon
from ecoscope_workflows_core.decorators import task
from selenium.webdriver.chrome.options import Options
from ecoscope_workflows_ext_ecoscope.tasks import analysis
from ecoscope_workflows_core.indexes import CompositeFilter
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerClient
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import ExportArgs
from ecoscope_workflows_ext_ecoscope.tasks.analysis._summary import SummaryParam
from ecoscope_workflows_core.skip import SkippedDependencyFallback, SkipSentinel
from typing import Sequence, Tuple, Union, Annotated, cast, Optional, Dict, List, Literal
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AdvancedField, AnyDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.analysis import calculate_elliptical_time_density

logger = logging.getLogger(__name__)


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

def normalize_file_url(path: str) -> str:
    """Convert file:// URL to local path, handling malformed Windows URLs."""
    if not path.startswith("file://"):
        return path

    path = path[7:]
    
    if os.name == 'nt':
        # Remove leading slash before drive letter: /C:/path -> C:/path
        if path.startswith('/') and len(path) > 2 and path[2] in (':', '|'):
            path = path[1:]

        path = path.replace('/', '\\')
        path = path.replace('|', ':')
    else:
        if not path.startswith('/'):
            path = '/' + path
    return path

def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 data URI."""
    try:
        path = Path(image_path).resolve()
        if not path.exists():
            logger.error(f"Image file not found: {image_path}")
            return ""
        
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
        ext = path.suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp',
        }.get(ext, 'image/png')
        
        return f'data:{mime_type};base64,{encoded}'
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return ""

@task
def get_area_bounds(
    gdf: AnyGeoDataFrame,
    names: Union[str, Sequence[str]],
    column: str,
) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) for all rows where `gdf[column]` matches any of `names`"""

    if column not in gdf.columns:
        raise KeyError(f"`get_area_bounds`:Column '{column}' not found. Available: {list(gdf.columns)}")

    if isinstance(names, str):
        names = [names]
    if not names:
        raise ValueError("`get_area_bounds`:`names` column must not be empty.")
    subset = gdf[gdf[column].isin(names)]

    if subset.empty:
        raise ValueError(f"`get_area_bounds`:No match found for {names} in column '{column}'.")

    xmin, ymin, xmax, ymax = subset.total_bounds
    return xmin, ymin, xmax, ymax

# doesnt exist on workflows yet
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
    subject_df: AnyDataFrame,
    column: str,
    output_path: Union[str, Path] = None,
    image_type: str = ".png",
    overwrite_existing: bool = True,
) -> Optional[str]:

    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = normalize_file_url(output_path)
    output_path = Path(output_path) 
    output_path.mkdir(parents=True, exist_ok=True) 

    if column not in subject_df.columns:
        raise KeyError(f"`download_profile_photo`: Column '{column}' not found. Available: {list(subject_df.columns)}") 

    for idx, url in subject_df[column].dropna().items():
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            logger.warning(f"Skipping invalid URL at index {idx}: {url}")
            continue

        try:
            df_subset = subject_df.loc[[idx]]
            row_hash = hashlib.sha256(pd.util.hash_pandas_object(df_subset, index=True).values).hexdigest()
            filename = f"{row_hash[:8]}_{idx}"

            if not image_type.startswith("."):
                image_type = f".{image_type}"

            file_path = output_path / f"{filename}{image_type}"
            processed_url = url.replace("dl=0", "dl=1") if "dropbox.com" in url else url
            ecoscope.io.utils.download_file(processed_url, str(file_path), overwrite_existing)
            logger.info(f"Downloaded profile photo for index {idx} to {file_path}")
        except Exception as e:
            logger.error(f"Error processing URL at index {idx} ({url}): {e}") 
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
def generate_subject_info(
    subject_df: AnyDataFrame,
    output_path: Union[str, Path],
    maxlen: int = 1000,
    return_data: bool = True,
) -> Optional[AnyDataFrame]:

    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()
    output_path = normalize_file_url(output_path)
    os.makedirs(output_path,exist_ok=True)
    
    if subject_df.empty:
        raise ValueError("`generate_subject_info`:DataFrame is empty")
    
    required_columns = [
        "subject_name","subject_bio","date_of_birth",
        "subject_sex","notes","country","distribution",
        "status",
    ]

    missing = [c for c in required_columns if c not in subject_df.columns]
    if missing:
        raise KeyError(
            f"Required columns {missing} don't exist in the dataframe. "
            f"Available columns: {list(subject_df.columns)}"
        )

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

    # Process all rows into a list of dicts
    processed_records = [process_single_subject(row) for _, row in subject_df.iterrows()]
    return cast(AnyDataFrame, pd.DataFrame(processed_records))

@task
def split_gdf_by_column(
    gdf: Annotated[AnyGeoDataFrame, Field(description="The GeoDataFrame to split")],
    column: Annotated[str, Field(description="Column name to split GeoDataFrame by")],
) -> Dict[str, AnyGeoDataFrame]:
    """
    Splits a GeoDataFrame into a dictionary of GeoDataFrames based on unique values in the specified column.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`split_gdf_by_column`:gdf is empty.")

    if column not in gdf.columns:
        raise ValueError(f"`split_gdf_by_column`:Column '{column}' not found in GeoDataFrame.")

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
        raise ValueError("`generate_mcp_gdf`:gdf is empty.")
    if gdf.geometry is None:
        raise ValueError("`generate_mcp_gdf`:gdf has no 'geometry' column.")
    if gdf.crs is None:
        raise ValueError("`generate_mcp_gdf`:gdf must have a CRS set (e.g., EPSG:4326).")

    original_crs = gdf.crs
    valid_points_gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    if valid_points_gdf.empty:
        raise ValueError("`generate_mcp_gdf`:No valid geometries in gdf.")
    
    projected_gdf = valid_points_gdf.to_crs(planar_crs)
    if not all(projected_gdf.geometry.geom_type.isin(["Point"])):
        projected_gdf.geometry = projected_gdf.geometry.centroid

    convex_hull = projected_gdf.geometry.unary_union.convex_hull

    area_sq_meters = float(convex_hull.area)
    area_sq_km = area_sq_meters / 1_000_000.0
    convex_hull_original_crs = gpd.GeoSeries([convex_hull], crs=planar_crs).to_crs(original_crs).iloc[0]

    result_gdf = gpd.GeoDataFrame(
        {
            "area_m2": [area_sq_meters], 
            "area_km2": [area_sq_km], 
            "mcp": "mcp"
        },
        geometry=[convex_hull_original_crs],
        crs=original_crs,
    )
    return result_gdf

@task
def create_seasonal_labels(
    traj: AnyGeoDataFrame, 
    total_percentiles: AnyDataFrame
    ) -> Optional[AnyGeoDataFrame]:
    """
    Annotates trajectory segments with seasonal labels (wet/dry) based on NDVI-derived windows.
    Applies to the entire trajectory without grouping.
    """
    try:
        if traj is None or traj.empty:
            raise ValueError("`create_seasonal_labels`:traj gdf is empty.")
        if total_percentiles is None or total_percentiles.empty:
            raise ValueError("`create_seasonal_labels `:total_percentiles df is empty.")

        seasonal_wins = total_percentiles.copy()
        traj_start = traj["segment_start"].min()
        traj_end = traj["segment_end"].max()

        seasonal_wins = seasonal_wins[
            (seasonal_wins["end"] >= traj_start) & (seasonal_wins["start"] <= traj_end)
        ].reset_index(drop=True)

        logger.info(f"Filtered seasonal windows: {len(seasonal_wins)} periods")
        logger.info(f"Seasonal Windows:\n{seasonal_wins[['start', 'end', 'season']]}")

        if seasonal_wins.empty:
            logger.error("No seasonal windows overlap with trajectory timeframe.")
            traj["season"] = None
            return traj

        season_bins = pd.IntervalIndex(data=seasonal_wins.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1))
        logger.info(f"Created {len(season_bins)} seasonal bins")

        labels = seasonal_wins["season"].values
        traj["season"] = pd.cut(traj["segment_start"], bins=season_bins, include_lowest=True).map(
            dict(zip(season_bins, labels))
        )
        null_count = traj["season"].isnull().sum()
        if null_count > 0:
            logger.warning(f"Warning: {null_count} trajectory segments couldn't be assigned to any season")

        logger.info("Seasonal labeling complete. Season distribution:")
        logger.info(traj["season"].value_counts(dropna=False))
        return traj
    except Exception as e:
        logger.error(f"Failed to apply seasonal label to trajectory: {e}")
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
        fillcolor = {"wet": "rgba(0,0,255,0.15)", "dry": "rgba(0,0,255,0.05)"}.get(season_type, "rgba(0,0,0,0.05)")

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
    # Check if it's already a DataFrame
    if isinstance(seasons_df, pd.DataFrame):
        return seasons_df
    
    # Normalize the path and convert to Path object
    normalized_path = normalize_file_url(str(seasons_df))
    p = Path(normalized_path)
    
    # testing 
    print(f"`_load_seasons_df`:path:{p}")
    
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    elif p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    else:
        return pd.read_csv(p)

@task
def generate_seasonal_nsd_plot(
    relocations_gdf: AnyGeoDataFrame,
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

    if relocations_gdf is None or relocations_gdf.empty:
        raise ValueError("`generate_seasonal_nsd_plot`:Relocations gdf is empty.")

    seasons_df = _load_seasons_df(seasons_df)
    relocations_gdf = Relocations.from_gdf(relocations_gdf)
    figure = nsd(relocations_gdf)
    figure = add_seasons_square(figure, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

@task
def generate_seasonal_speed_plot(
    relocations_gdf: AnyGeoDataFrame,
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

    if relocations_gdf is None or relocations_gdf.empty:
        raise ValueError("`generate_seasonal_speed_plot`:Relocations gdf is empty.")

    seasons_df = _load_seasons_df(seasons_df)
    relocations_gdf = Relocations.from_gdf(relocations_gdf)
    trajs_gdf = Trajectory.from_relocations(relocations_gdf)
    figure = ecoscope.plotting.speed(trajs_gdf)
    figure = add_seasons_square(figure, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

@task
def generate_seasonal_mcp_asymptote_plot(
    relocations_gdf: AnyGeoDataFrame,
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
    if relocations_gdf is None or relocations_gdf.empty:
        raise ValueError("`generate_seasonal_mcp_asymptote_plot`:Relocations gdf is empty.")

    seasons_df = _load_seasons_df(seasons_df)
    relocations_gdf = Relocations.from_gdf(relocations_gdf)
    figure = ecoscope.plotting.mcp(relocations_gdf)
    figure = add_seasons_square(figure, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

def collar_event_timeline_plot(
    geodataframe: AnyGeoDataFrame,
    collar_events: Optional[AnyDataFrame] = None,
) -> go.FigureWidget:

    fig = go.FigureWidget()
    if  geodataframe is None or geodataframe.empty:
        raise ValueError("`collar_event_timeline_plot`:Relocations gdf is empty.")

    required_cols = ["fixtime"]
    missing_cols = [col for col in required_cols if col not in geodataframe.columns]
    if missing_cols:
        raise ValueError(f"`collar_event_timeline_plot`:Missing required columns: {missing_cols}")

    geodataframe = geodataframe.dropna(subset=["fixtime"]).copy()
    
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

    if relocations_gdf is None or relocations_gdf.empty:
        raise ValueError("`generate_collared_seasonal_plot`:Relocations gdf is empty.")

    subject_name = relocations_gdf['subject_name'].unique()[0]

    if events_gdf is None or events_gdf.empty:
        logger.warning(f"`generate_collared_seasonal_plot`:No events data for subject '{subject_name}'.")
        events_gdf = None
    elif filter_col not in events_gdf.columns:
        raise ValueError(f"`generate_collared_seasonal_plot`:Column '{filter_col}' not found. Available: {', '.join(events_gdf.columns)}")
    else:
        events_gdf = events_gdf[events_gdf[filter_col] == subject_name]
        if events_gdf.empty:
            logger.warning(f"`generate_collared_seasonal_plot`:No events found for subject '{subject_name}'.")
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

    if relocations_gdf is None or relocations_gdf.empty:
        raise ValueError("`compute_maturity`:Relocations gdf is empty.")
    if "groupby_col" not in relocations_gdf.columns:
        raise ValueError("`compute_maturity`:Relocations gdf must contain 'groupby_col' column.")
    if time_column not in relocations_gdf.columns:
        raise ValueError(f"`compute_maturity`:Relocations gdf must contain '{time_column}' column.")

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

@task
def get_subject_stats(
    traj_gdf: AnyGeoDataFrame,
    subject_df: AnyDataFrame,
    etd_df: AnyGeoDataFrame,
    groupby_col: str,
    output_path: Union[str, Path]=None,
) ->AnyDataFrame:
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = normalize_file_url(output_path)

    if traj_gdf is None or traj_gdf.empty:
        raise ValueError("`get_subject_stats`:`traj_gdf` is empty.")

    if "geometry" not in traj_gdf.columns:
        raise ValueError("`get_subject_stats`:`traj_gdf` must have a geometry column.")

    if groupby_col not in traj_gdf.columns:
        raise ValueError(f"`get_subject_stats`:`traj_gdf` must contain '{groupby_col}'.")
        
    if "segment_start" not in traj_gdf.columns:
        raise ValueError("`get_subject_stats`:`traj_gdf` must contain a 'segment_start' datetime column.")

    # Parse datetime column
    traj_gdf = traj_gdf.copy()
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

    # Calculate ETD area
    etd_km2 = 0.0
    if etd_df is not None and not etd_df.empty:
        if groupby_col in etd_df.columns:
            etd_sub = etd_df.loc[etd_df[groupby_col] == subject_id]
            if not etd_sub.empty and "area" in etd_sub.columns:
                etd_km2 = round(float(etd_sub["area"].sum()), 1)

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
    first_geom = traj_gdf.geometry.iloc[0]
    max_displacement_km = round(
        float(traj_gdf.geometry.distance(first_geom).max()) / 1000.0, 1
    )

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
        raise ValueError("`calculate_seasonal_home_range`:gdf is empty.")
    
    if 'season' not in gdf.columns:
        raise ValueError("`calculate_seasonal_home_range`:gdf must have a 'season' column.")
    
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
        raise ValueError("`build_template_region_lookup`:gdf cannot be empty.")

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
    regions_gdf: Dict[str, Union[Polygon, MultiPolygon]],
    output_path: Union[str, Path],
) -> AnyDataFrame:
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = normalize_file_url(output_path)
    if subjects_df is None or subjects_df.empty:
        raise ValueError("`compute_subject_occupancy`:Subjects dataframe is empty.")
    if etd_gdf is None or etd_gdf.empty:
        raise ValueError("`compute_subject_occupancy`:ETD GeoDataFrame is empty.")
    if regions_gdf is None or not regions_gdf:
        raise ValueError("`compute_subject_occupancy`:Regions dictionary is empty.")
    
    subject_id = subjects_df['subject_name'].iloc[0]
    
    # Get home range at 99.9th percentile and convert to target CRS
    try:
        percentile_mask = etd_gdf['percentile'] == 99.9
        if not percentile_mask.any():
            raise ValueError(
                f"`compute_subject_occupancy`:No 99.9th percentile found for subject '{subject_id}'. "
                f"Available percentiles: {sorted(etd_gdf['percentile'].unique().tolist())}"
            )
        
        subject_range = etd_gdf[percentile_mask].to_crs(crs).geometry.iloc[0]
        
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"`compute_subject_occupancy`:Could not extract 99.9th percentile ETD for subject '{subject_id}'. "
            f"Available percentiles: {etd_gdf['percentile'].unique().tolist()}"
        ) from e
    
    if subject_range.is_empty:
        raise ValueError(f"`compute_subject_occupancy`:Home range geometry is empty for subject '{subject_id}'.")
    total_area = subject_range.area
    if total_area == 0:
        raise ValueError(f"`compute_subject_occupancy`:Home range has zero area for subject '{subject_id}'.")
    occupancy = {}
    for region_name, region_geom in regions_gdf.items():
        try:
            intersection_area = region_geom.intersection(subject_range).area
            occupancy[region_name] = 100 * (intersection_area / total_area)
        except Exception as e:
            logger.error(f"Warning: Failed to compute intersection for region '{region_name}': {e}")
            occupancy[region_name] = 0.0

    # Calculate unprotected area
    national_pa = occupancy.get("national_pa_use", 0.0)
    community_pa = occupancy.get("community_pa_use", 0.0)
    occupancy["unprotected"] = max(0.0, 100.0 - national_pa - community_pa)
    occupancy = {k: round(v, 1) for k, v in occupancy.items()}
    return cast(AnyDataFrame, pd.DataFrame([occupancy]))

@task
def download_file_and_persist(
    url: Annotated[str, Field(description="URL to download the file from")],
    output_path: Annotated[
        Optional[str], 
        Field(
            description="Path to save the downloaded file or directory. Defaults to current working directory")
            ] = None,
    retries: Annotated[int, Field(description="Number of retries on failure", ge=0)] = 3,
    overwrite_existing: Annotated[bool, Field(description="Whether to overwrite existing files")] = False,
    unzip: Annotated[bool, Field(description="Whether to unzip the file if it's a zip archive")] = False,
) -> str:
    """
    Downloads a file from the provided URL and persists it locally.
    If output_path is not specified, saves to the current working directory.
    Returns the full path to the downloaded file, or if unzipped, the path to the extracted directory.
    """
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = normalize_file_url(output_path)
    looks_like_dir = (
        output_path.endswith(os.sep)
        or output_path.endswith("/")
        or output_path.endswith("\\")
        or os.path.isdir(output_path)
    )

    if looks_like_dir:
        # ensure directory exists
        os.makedirs(output_path, exist_ok=True)

        # determine filename from Content-Disposition or URL
        import requests, email
        try:
            s = requests.Session()
            r = s.head(url, allow_redirects=True, timeout=10)
            cd = r.headers.get("content-disposition", "")
            filename = None
            if cd:
                # parse content-disposition safely
                m = email.message.Message()
                m["content-disposition"] = cd
                filename = m.get_param("filename")
            if not filename:
                filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"
        except Exception:
            filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"

        target_path = os.path.join(output_path, filename)
    else:
        target_path = output_path

    if not target_path or str(target_path).strip() == "":
        raise ValueError("Computed download target path is empty. Check 'output_path' argument.")

    # Store the parent directory to check for extracted content
    parent_dir = os.path.dirname(target_path)
    before_extraction = set()
    if unzip:
        if os.path.exists(parent_dir):
            before_extraction = set(os.listdir(parent_dir))

    # Do the download and bubble up useful context on failure
    try:
        download_file(
            url=url,
            path=target_path,
            retries=retries,
            overwrite_existing=overwrite_existing,
            unzip=unzip,
        )
    except Exception as e:
        # include debug info so callers can see what was attempted
        raise RuntimeError(
            f"download_file failed for url={url!r} path={target_path!r} retries={retries}. "
            f"Original error: {e}"
        ) from e

    # Determine the final persisted path
    if unzip and zipfile.is_zipfile(target_path):
        after_extraction = set(os.listdir(parent_dir))
        new_items = after_extraction - before_extraction
        zip_filename = os.path.basename(target_path)
        new_items.discard(zip_filename)
        
        if len(new_items) == 1:
            new_item = new_items.pop()
            new_item_path = os.path.join(parent_dir, new_item)
            if os.path.isdir(new_item_path):
                persisted_path = str(Path(new_item_path).resolve())
            else:
                persisted_path = str(Path(parent_dir).resolve())
        elif len(new_items) > 1:
            persisted_path = str(Path(parent_dir).resolve())
        else:
            extracted_dir = target_path.rsplit('.zip', 1)[0]
            if os.path.isdir(extracted_dir):
                persisted_path = str(Path(extracted_dir).resolve())
            else:
                persisted_path = str(Path(parent_dir).resolve())
    else:
        persisted_path = str(Path(target_path).resolve())

    if not os.path.exists(persisted_path):
        parent = os.path.dirname(persisted_path)
        if os.path.exists(parent):
            actual_files = os.listdir(parent)
            raise FileNotFoundError(
                f"Download failed — {persisted_path} not found after execution. "
                f"Files in {parent}: {actual_files}"
            )
        else:
            raise FileNotFoundError(
                f"Download failed — {persisted_path}. Parent dir missing: {parent}"
            )
    return persisted_path

@task
def report_context(
    input_path: str,
    subjects_df: AnyDataFrame,
    movement_ecomap: str,
    range_ecomap: str,
    overview_map: str,
    nsd_plot: str,
    speed_plot: str,
    mcp_plot: str,
    collar_event_timeline_plot: str,
    logo: str,
    subject_photo: str,
    subject_info: str,
    subject_stats: str,
    occupancy_info: str,
    templates: list[str],
) -> list[str]:
    if input_path is None or str(input_path).strip() == "":
        input_path = os.getcwd()
    else:
        input_path = str(input_path).strip()

    input_path = normalize_file_url(input_path)

    # ensure all paths are normalized
    movement_ecomap =normalize_file_url(movement_ecomap)
    range_ecomap =normalize_file_url(range_ecomap)
    overview_ecomap =normalize_file_url(overview_ecomap)
    nsd_plot =normalize_file_url(nsd_plot)
    speed_plot =normalize_file_url(speed_plot)
    collar_event_timeline_plot =normalize_file_url(collar_event_timeline_plot)
    logo =normalize_file_url(logo)
    subject_photo =normalize_file_url(subject_photo)
    subject_info =normalize_file_url(subject_info)
    subject_stats =normalize_file_url(subject_stats)
    occupancy_info = normalize_file_url(occupancy_info)

    for template in templates:
        template = normalize_file_url(template)

    if subjects_df is None or subjects_df.empty:
        raise ValueError("`report_context`:Subjects df is empty.")
    
    required_cols = ["mature", "subject_name"]
    missing_cols = [col for col in required_cols if col not in subjects_df.columns]
    if missing_cols:
        raise ValueError(f"`report_context`:Missing required columns: {missing_cols}")
        
    # Set template environment
    template_loader = jinja2.FileSystemLoader(searchpath=input_path)
    template_env = jinja2.Environment(loader=template_loader)

    # Initialize context dictionary
    context = {}
    
    # Load CSV files into DataFrames
    try:
        subject_info_df = pd.read_csv(subject_info)
    except Exception as e:
        raise ValueError(f"`report_context`:Failed to load subject_info CSV from '{subject_info}': {e}")
    
    try:
        subject_stats_df = pd.read_csv(subject_stats)
    except Exception as e:
        raise ValueError(f"`report_context`:Failed to load subject_stats CSV from '{subject_stats}': {e}")
    
    try:
        occupancy_info_df = pd.read_csv(occupancy_info)
    except Exception as e:
        raise ValueError(f"`report_context`:Failed to load occupancy_info CSV from '{occupancy_info}': {e}")
    
    # Convert individual data DataFrames to dictionaries
    if not subject_info_df.empty:
        subject_info_dict = subject_info_df.iloc[0].to_dict()
        context.update(subject_info_dict)
    else:
        raise ValueError("`report_context`:subject_info df is empty")
    
    if not subject_stats_df.empty:
        subject_stats_dict = subject_stats_df.iloc[0].to_dict()
        context.update(subject_stats_dict)
    else:
        logger.warning("`report_context`: subject_stats df is empty")
    
    if not occupancy_info_df.empty:
        occupancy_dict = occupancy_info_df.iloc[0].to_dict()
        context["occupancy"] = occupancy_dict
    else:
        logger.warning("`report_context`: occupancy_info df is empty")
    
    # Get subject-specific data
    subject_name = context.get("subject_name")
    if not subject_name:
        raise ValueError("`report_context`:subject_name not found in subject_info")
    
    subject_row = subjects_df[subjects_df['subject_name'] == subject_name]
    if subject_row.empty:
        raise ValueError(f"Subject '{subject_name}' not found in subjects_df")
    
    mature = bool(subject_row['mature'].iloc[0])
    context["mature"] = mature
    
    logger.info(f"\n=== Processing images for {subject_name} ===")
    
    # Add logo with base64 encoding
    logo_path = Path(logo).resolve()
    if logo_path.exists():
        logo_base64 = encode_image_to_base64(str(logo_path))
        if logo_base64:
            context["logo_path"] = f'<img src="{logo_base64}" alt="Logo" style="max-height: 100px;"/>'
            logger.info(f"Logo encoded successfully")
        else:
            context["logo_path"] = ""
            logger.error(f"Logo encoding failed")
    else:
        logger.error(f"Logo not found at {logo}")
        context["logo_path"] = ""
    
    # Add subject photo with base64 encoding
    subject_photo_path = Path(subject_photo).resolve()
    if subject_photo_path.exists():
        photo_base64 = encode_image_to_base64(str(subject_photo_path))
        if photo_base64:
            context["id_photo"] = f'<img src="{photo_base64}" alt="Subject Photo" style="max-width: 100%;"/>'
            logger.info(f"Subject photo encoded successfully")
        else:
            context["id_photo"] = "<div>Photo not available</div>"
            logger.error(f"Subject photo encoding failed")
    else:
        logger.error(f"Subject photo not found at {subject_photo}")
        context["id_photo"] = "<div>Photo not available</div>"
    
    # Add plot images to context with base64 encoding
    plot_mapping = {
        "mov_map": movement_ecomap,
        "nsd_plot": nsd_plot,
        "speed_plot": speed_plot,
        "mcp_plot": mcp_plot,
        "collar_event_timeline": collar_event_timeline_plot,
    }
    
    for key, path in plot_mapping.items():
        path_obj = Path(path).resolve()
        if path_obj.exists():
            img_base64 = encode_image_to_base64(str(path_obj))
            if img_base64:
                context[key] = f'<img style="width:100%;height:100%;object-fit:cover;" src="{img_base64}"/>'
                logger.info(f"{key} encoded successfully")
            else:
                context[key] = "<div>Image not found</div>"
                logger.error(f"{key} encoding failed")
        else:
            logger.error(f"{key} not found at {path}")
            context[key] = "<div>Image not found</div>"
    
    # Add range ecomap with base64 encoding
    range_path = Path(range_ecomap).resolve()
    if range_path.exists():
        range_base64 = encode_image_to_base64(str(range_path))
        if range_base64:
            context["range_map"] = f'<img style="max-width:100%;max-height:100%;" src="{range_base64}"/>'
            logger.info(f"Range map encoded successfully")
        else:
            context["range_map"] = "<div>Range map not found</div>"
            logger.error(f"Range map encoding failed")
    else:
        logger.error(f"Range map not found at {range_ecomap}")
        context["range_map"] = "<div>Range map not found</div>"
    
    # Add overview map with base64 encoding
    overview_path = Path(overview_map).resolve()
    if overview_path.exists():
        overview_base64 = encode_image_to_base64(str(overview_path))
        if overview_base64:
            context["overview_map"] = f'<img style="max-width:100%;max-height:100%;" src="{overview_base64}"/>'
            logger.info(f"Overview map encoded successfully")
        else:
            context["overview_map"] = "<div>Overview map not found</div>"
            logger.error(f"Overview map encoding failed")
    else:
        logger.error(f"Overview map not found at {overview_map}")
        context["overview_map"] = "<div>Overview map not found</div>"
    
    # Determine which templates to use based on maturity
    if mature:
        template_files = templates  # Use all templates
    else:
        # Exclude template_3.html for immature subjects
        template_files = [t for t in templates if "template_3.html" not in t]
    
    logger.info(f"\n=== Rendering templates for {subject_name} ===")
    logger.info(f"Templates to render: {len(template_files)}")
    
    #testing 
    print(f"context info: {context}")
    
    # Render templates
    rendered_files = []
    for template_file in template_files:
        try:
            template_filename = Path(template_file).name
            template = template_env.get_template(template_filename)
            
            # Create output filename based on template and subject
            template_name = Path(template_file).stem
            output_filename = f"{subject_name}_{template_name}.html"
            output_file_path = output_dir / output_filename
            
            # Render and save
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(template.render(**context))
            
            logger.info(f"Rendered: {output_filename}")
            rendered_files.append(str(output_file_path))
            
        except jinja2.TemplateNotFound:
            logger.error(f"Template not found: {template_file}")
        except Exception as e:
            logger.error(f"Failed to render {template_file}: {e}")
    
    logger.info(f"\nTotal files rendered: {len(rendered_files)}")
    return rendered_files

@task
def render_html_to_pdf(html_path: Union[str, list[str]], pdf_path: str) -> List[str]:
    """Render HTML file(s) to PDF and return list of PDF paths."""
    # Normalize input
    html_paths = [html_path] if isinstance(html_path, str) else list(html_path or [])
    if not html_paths:
        raise ValueError("`render_html_to_pdf`:No HTML paths provided")
    for html_path in html_paths:
        html_path = normalize_file_url(html_path)

    if pdf_path is None or str(pdf_path).strip() == "":
        pdf_path = os.getcwd()
    else:
        pdf_path = str(pdf_path).strip()

    pdf_path = normalize_file_url(pdf_path)
    os.makedirs(pdf_path, exist_ok=True)

    # Chrome options (allow local file access)
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--allow-file-access-from-files")
    chrome_options.add_argument("--allow-file-access")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--enable-local-file-accesses")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)

    cmd_args = {
        "printBackground": True,
        "preferCSSPageSize": True,
    }

    # # Normalize pdf_path (could be file:// URI)
    # if pdf_path.startswith("file://"):
    #     parsed = urlparse(pdf_path)
    #     pdf_path = url2pathname(parsed.path)
    
    # os.makedirs(pdf_path, exist_ok=True)
    pdf_paths = []

    try:
        for html_file in html_paths:
            # html_file = str(html_file)
            
            # # Normalize html_file if it's a file:// URI
            # if html_file.startswith("file://"):
            #     parsed = urlparse(html_file)
            #     html_file = url2pathname(parsed.path)
            
            if not os.path.exists(html_file):
                raise FileNotFoundError(f"`render_html_to_pdf`:HTML file not found: {html_file}")

            if not html_file.lower().endswith(".html"):
                raise ValueError(f"`render_html_to_pdf`:Input file must be an HTML file: {html_file}")

            # Generate output path - use basename only, no directory info from html_file
            pdf_filename = Path(html_file).stem + ".pdf"
            output_pdf_path = os.path.join(pdf_path, pdf_filename)
            
            # Use file URI for loading in browser
            html_uri = Path(html_file).resolve().as_uri()
            logger.info(f"[debug] Loading HTML URI: {html_uri}")

            driver.get(html_uri)

            # Wait for full load
            max_wait = 10
            waited = 0
            while waited < max_wait:
                ready = driver.execute_script("return document.readyState")
                if ready == "complete":
                    break
                time.sleep(0.5)
                waited += 0.5
            else:
                logger.error("[warning] document.readyState did not reach 'complete' within timeout")

            time.sleep(0.5)

            # Generate PDF
            result = driver.execute_cdp_cmd("Page.printToPDF", cmd_args)
            pdf_data = result.get("data")
            if not pdf_data:
                raise RuntimeError("No PDF data returned from Chrome DevTools Protocol")

            with open(output_pdf_path, "wb") as f:
                f.write(base64.b64decode(pdf_data))

            if os.path.exists(output_pdf_path):
                size = os.path.getsize(output_pdf_path)
                logger.info(f"PDF saved: {output_pdf_path} (size={size} bytes)")
            else:
                logger.error(f"WARNING: failed to write PDF: {output_pdf_path}")

            # Return absolute path
            pdf_paths.append(os.path.abspath(output_pdf_path))

        logger.info(f"PDF generation complete. Files: {pdf_paths}")
        return pdf_paths

    except Exception as e:
        logger.error(f"Failed to render HTML to PDF: {e}")
        raise
    finally:
        try:
            driver.quit()
        except Exception:
            pass
            
def _fallback_to_none_doc(
    obj: tuple[CompositeFilter | None, str] | SkipSentinel
    ) -> tuple[CompositeFilter | None, str] | None:
    return None if isinstance(obj, SkipSentinel) else obj

@dataclass
class GroupedDoc:
    """Analogous to GroupedWidget but for document pages."""
    views: dict[CompositeFilter | None, Optional[str]]

    @classmethod
    def from_single_view(cls, item: tuple[CompositeFilter | None, str]) -> "GroupedDoc":
        view, path = item
        return cls(views={view: path})

    @property
    def merge_key(self) -> str:
        """
        Determine how docs should be grouped.
        Default: group by filename stem of the first non-None path in views.
        """
        for p in self.views.values():
            if p:
                return Path(p).stem
        return uuid.uuid4().hex

    def __ior__(self, other: "GroupedDoc") -> "GroupedDoc":
        """Merge views from other into self. Keys must be compatible by merge_key."""
        if self.merge_key != other.merge_key:
            raise ValueError(
                f"Cannot merge GroupedDoc with different keys: {self.merge_key} != {other.merge_key}"
            )
        self.views.update(other.views)
        return self


@task
def merge_pdfs(
    subject_df: AnyDataFrame,
    pdf_path_items: Annotated[
        Union[
            list[str],  # Accept plain list of strings
            list[Union[str, List[str]]],  # Accept nested lists
            list[tuple[CompositeFilter | None, Union[str, List[str]]]],  # Accept tuples with filters
        ],
        Field(description="List of PDF paths. Can be strings, lists, or tuples with filters.", exclude=True),
    ],
    output_path: str,
    filename: str
) -> str:
    """Merge all PDFs into a single document organized by subject using GroupedDoc pattern."""
    
    if subject_df is None or subject_df.empty:
        raise ValueError("`merge_pdfs`: subject_df is empty.")
    
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = normalize_file_url(output_path)
    os.makedirs(output_path,exist_ok=True)
    
    # Normalize input to tuples format
    normalized_items = []
    logger.info(f"pdf path items:{pdf_path_items}")
    for item in pdf_path_items:
        if item is None:
            continue
        elif isinstance(item, tuple):
            # Already in (filter, path) format
            normalized_items.append(item)
        else:
            # Plain string or list - wrap with None filter
            normalized_items.append((None, item))
    
    # Filter out None items
    valid_items = normalized_items
    logger.info(f"Valid items: {valid_items}")

    # Flatten any nested lists in the paths and create GroupedDoc instances
    flattened_items = []
    for view_key, paths in valid_items:
        # Handle both single paths and lists of paths
        if isinstance(paths, str):
            path_list = [paths]
        elif isinstance(paths, list):
            # Flatten nested lists
            path_list = []
            for item in paths:
                if isinstance(item, list):
                    path_list.extend(item)
                else:
                    path_list.append(item)
        else:
            path_list = []
        
        # Create a GroupedDoc item for each path
        for path in path_list:
            flattened_items.append((view_key, path))
    
    logger.info(f"Flattened items:{flattened_items}")

    # Create GroupedDoc instances
    grouped_docs = [GroupedDoc.from_single_view(it) for it in flattened_items]
    
    # Merge docs with same merge_key
    merged_map: dict[str, GroupedDoc] = {}
    for gd in grouped_docs:
        key = gd.merge_key
        if key not in merged_map:
            merged_map[key] = gd
        else:
            merged_map[key] |= gd
    
    # Extract all PDF paths from grouped docs
    all_pdf_paths: list[str] = []
    for group in merged_map.values():
        for view_key, p in group.views.items():
            if p is not None:
                all_pdf_paths.append(p)
    
    # Clean up file:// URIs and malformed paths
    cleaned_pdf_paths = []
    # for path in all_pdf_paths:
    #     path = str(path)
        
    #     # Handle file:// URIs
    #     if path.startswith("file://"):
    #         parsed = urlparse(path)
    #         path = url2pathname(parsed.path)
        
    #     # Fix malformed paths with 'file:' as directory component
    #     if "/file:/" in path or "\\file:\\" in path:
    #         parts = path.split("file:")
    #         if len(parts) > 1:
    #             path = parts[-1].lstrip("/\\")
    #             if not path.startswith("/"):
    #                 path = "/" + path
        
    #     cleaned_pdf_paths.append(path)
    
    for path in all_pdf_paths:
        path = normalize_file_url(path)
        cleaned_pdf_paths.append(path)
    
    logger.info(f"[debug] Cleaned PDF paths: {cleaned_pdf_paths}")
    
    # Verify all PDFs exist
    for pdf_file in cleaned_pdf_paths:
        if not os.path.exists(pdf_file):
            logger.warning(f"[warning] PDF file not found: {pdf_file}")
    
    merger = PdfMerger()
    subject_names = sorted(subject_df["subject_name"].unique())
    
    # Merge PDFs in order by subject
    for subject_name in subject_names:
        subject_pdfs = []
        
        for pdf_file in cleaned_pdf_paths:
            if pdf_file.endswith(".pdf"):
                pdf_basename = os.path.basename(pdf_file)
                if pdf_basename.startswith(subject_name):
                    if os.path.exists(pdf_file):
                        subject_pdfs.append(pdf_file)
        
        # Add subject's PDFs to merger
        if subject_pdfs:
            for pdf_file in sorted(subject_pdfs):
                try:
                    logger.info(f"[debug] Adding to merger: {pdf_file}")
                    merger.append(pdf_file)
                except Exception as e:
                    logger.error(f"[warning] Failed to add {pdf_file} to merger: {e}")
        else:
            logger.error(f"[warning] No PDFs found for subject: {subject_name}")
    
    output_filename = f"{filename}.pdf" if not filename.endswith(".pdf") else filename
    final_path = os.path.join(output_path, output_filename)
    
    with open(final_path, "wb") as f:
        merger.write(f)
    
    merger.close()
    
    if os.path.exists(final_path):
        size = os.path.getsize(final_path)
        logger.info(f"[success] Final merged PDF saved: {final_path} (size={size} bytes)")
    else:
        raise RuntimeError(f"Failed to create merged PDF: {final_path}")
    
    return final_path
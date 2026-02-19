import os
import logging
from pydantic import Field
from datetime import datetime
from typing import Annotated, Union, List
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.io import persist_text
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_ecoscope.connections import EarthEngineClient
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import (
    PlotStyle,
    LineStyle,
    AxisStyle,
    LayoutStyle,
    draw_historic_timeseries,
)
from ecoscope_workflows_ext_ecoscope.tasks.io._earthengine import calculate_ndvi_range


logger = logging.getLogger(__name__)

@task
def process_aoi_ndvi_charts(
    df: Annotated[
        AnyGeoDataFrame,
        Field(description="GeoDataFrame containing ranch data", exclude=True),
    ],
    er_client: Annotated[
        EarthEngineClient,
        Field(description="EarthEngine client connection", exclude=True),
    ],
   aoi_column: Annotated[str, Field(description="Column name containing ranch names")],
   time_range: Annotated[TimeRange, Field(description="Time range for NDVI analysis")],
    output_dir: Annotated[str, Field(description="Directory to save output HTML charts")],
) -> Union[str, List[str]]:
    # Validate ranch name column exists
    if aoi_column not in df.columns:
        available = ", ".join(list(df.columns)[:10])
        raise ValueError(f"Column '{aoi_column}' not found. Available columns: {available}...")

    # Get unique ranch names
    aoi_names = df[aoi_column].unique()
    logger.info(f"Processing NDVI charts for {len(aoi_names)} ranches")

    if output_dir is None or str(output_dir).strip() == "":
        output_dir = os.getcwd()

    file_paths = []

    # Ensure CRS is 4326 before processing
    if df.crs is None or df.crs.to_epsg() != 4326:
        df = df.to_crs(4326)
    
    # Process each ranch
    for idx, aoi_name in enumerate(aoi_names, 1):
        logger.info(f"Processing ranch {idx}/{len(aoi_names)}: {aoi_name}")

        try:
            aoi = df[df[aoi_column] == aoi_name].copy()

            if aoi.empty:
                logger.warning(f"No data found for ranch: {aoi_name}")
                continue

            if not aoi.geometry.is_valid.all():
                logger.warning(f"Invalid geometries found for ranch: {aoi_name}, attempting repair")
                aoi["geometry"] = aoi.geometry.buffer(0)

                if not aoi.geometry.is_valid.all():
                    logger.error(f"Could not repair geometries for ranch: {aoi_name}")
                    continue

            # Calculate NDVI range
            dfs = calculate_ndvi_range(
                client=er_client,
                roi=aoi,
                time_range=time_range,
                img_coll_name="MODIS/061/MYD13A1",
                band="NDVI",
                scale_factor=0.0001,
                analysis_scale=500.0,
            )
    
            # Create historic timeseries chart
            ndvi_chart = draw_historic_timeseries(
                dataframe=dfs,
                current_value_column="NDVI",
                current_value_title="Current NDVI",
                historic_min_column="min",
                historic_max_column="max",
                historic_band_title="Historic Range",
                historic_mean_column="mean",
                historic_mean_title="Historic Mean",
                layout_style=LayoutStyle(
                    title=f"{aoi_name}",
                    title_x=0.5,
                    font_size=14,
                    font_color="#222222",
                    plot_bgcolor="#f5f5f5",
                    showlegend=True,
                    hovermode="x unified",
                    xaxis=AxisStyle(title="Date"),
                    yaxis=AxisStyle(
                        title="NDVI",
                        range=[0.0, 1.0],
                    ),
                ),
                # Historic min–max band
                upper_lower_band_style=PlotStyle(
                    mode="lines",
                    fillcolor="rgba(76, 175, 80, 0.25)",
                    line=LineStyle(
                        color="rgba(76, 175, 80, 0.4)",
                        dash="dot",
                    ),
                ),
                # Historic mean
                historic_mean_style=PlotStyle(
                    mode="lines",
                    line=LineStyle(
                        color="#43A047",
                        dash="dash",
                    ),
                ),
                # Current NDVI
                current_value_style=PlotStyle(
                    mode="lines",
                    line=LineStyle(
                        color="#1B5E20",
                    ),
                ),
            )
            safe_name = str(aoi_name).replace(" ", "_").replace("/", "_")
            file_path = persist_text(ndvi_chart, str(output_dir), f"{safe_name}.html")

            file_paths.append(file_path)
            logger.info(f"Saved chart for {aoi_name} to {file_path}")

        except Exception as e:
            logger.error(f"Failed to process ranch {aoi_name}: {e}")
            continue

    return file_paths

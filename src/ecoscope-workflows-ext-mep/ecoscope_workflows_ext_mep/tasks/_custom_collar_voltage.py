import os
import logging
import pandas as pd
from pydantic import Field
from typing import Annotated, List
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.io import persist_text
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.tasks import analysis, transformation
from ecoscope_workflows_core.tasks.analysis._aggregation import (
dataframe_column_first_unique,
dataframe_column_percentile,
dataframe_column_mean,
apply_arithmetic_operation,
dataframe_column_min,
dataframe_column_max)

from ecoscope_workflows_core.tasks.transformation._extract import FieldType
from ecoscope_workflows_core.tasks.transformation._filter import ComparisonOperator
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import (
    AxisStyle,
    LayoutStyle,
    LineStyle,
    PlotStyle,
    draw_historic_timeseries,
)

logger = logging.getLogger(__name__)


@task
def process_collar_voltage_charts(
    relocs: Annotated[
        AnyGeoDataFrame,
        Field(description="The relocation geodataframe for current period.", exclude=True)
    ],
    time_range: Annotated[TimeRange, Field(description="Time range for voltage analysis")],
    output_dir: Annotated[str, Field(description="Directory to save output HTML charts")],
    previous_relocs: Annotated[
        AnyGeoDataFrame | None,
        Field(
            default=None,
            description="Optional relocation geodataframe for previous period to use for historical baseline calculations. If None, uses data before time_range.since from relocs.",
            exclude=True
        )
    ] = None,
) -> List[str]:
    """
    Generate collar voltage charts for all unique subjects in relocation data.
    
    Args:
        relocs: GeoDataFrame containing relocation data with voltage information for current period
        time_range: TimeRange object specifying analysis period
        output_dir: Directory to save generated HTML charts
        previous_relocs: Optional GeoDataFrame for previous period data to calculate historical baselines.
                        If not provided, uses historical data from before time_range.since in relocs.
        
    Returns:
        List of file paths for successfully generated charts
    """

    if output_dir is None or str(output_dir).strip() == "":
        output_dir = os.getcwd()
    
    # Extract columns for current relocations
    relocs = transformation.extract_column_as_type(
        relocs, 
        "extra__subjectsource__assigned_range", 
        FieldType.SERIES, 
        "extra.extra.subjectsource__assigned_range."
    )
    relocs = transformation.extract_column_as_type(
        relocs,
        "extra.extra.subjectsource__assigned_range.upper",
        FieldType.DATETIME,
        "extra.extra.subjectsource__assigned_range.upper",
    )

    # If previous_relocs provided, extract columns for it too
    if previous_relocs is not None:
        previous_relocs = transformation.extract_column_as_type(
            previous_relocs, 
            "extra__subjectsource__assigned_range", 
            FieldType.SERIES, 
            "extra.extra.subjectsource__assigned_range."
        )
        previous_relocs = transformation.extract_column_as_type(
            previous_relocs,
            "extra.extra.subjectsource__assigned_range.upper",
            FieldType.DATETIME,
            "extra.extra.subjectsource__assigned_range.upper",
        )

    groups = relocs.groupby(by=["extra__subject__name", "extra__subjectsource__id"])

    logger.info(f"Processing voltage charts for {len(groups)} subjects")
    print(f"Processing voltage charts for {len(groups)} subjects")
    
    # Process each group
    file_paths = []
    skipped_count = 0
    processed_count = 0

    for (subject_name, subjectsource_id), dataframe in groups:
        logger.info(f"Processing subject: {subject_name} (ID: {subjectsource_id})")
        
        try:
            subjectsource_upperbound = dataframe_column_first_unique(
                dataframe, "extra.extra.subjectsource__assigned_range.upper"
            )

            if not pd.isna(subjectsource_upperbound) and subjectsource_upperbound < pd.to_datetime(time_range.since):
                logger.info(f"Skipping {subject_name}: collar deactivated before time range")
                skipped_count += 1
                continue

            dataframe = transformation.sort_values(dataframe, "fixtime")
            dataframe = transformation.extract_value_from_json_column(
                dataframe,
                "extra__observation_details",
                ["battery", "mainVoltage", "batt", "power"],
                FieldType.FLOAT,
                "voltage",
            )

            # Filter current period data
            curr_df = transformation.filter_df(dataframe, "fixtime", ComparisonOperator.GE, time_range.since)
            curr_df = transformation.filter_df(curr_df, "fixtime", ComparisonOperator.LT, time_range.until)

            # Get historical data - either from previous_relocs or from before time_range.since
            if previous_relocs is not None:
                # Use previous relocations for historical baseline
                subject_prev_data = previous_relocs[
                    (previous_relocs["extra__subject__name"] == subject_name) & 
                    (previous_relocs["extra__subjectsource__id"] == subjectsource_id)
                ].copy()
                
                if not subject_prev_data.empty:
                    subject_prev_data = transformation.sort_values(subject_prev_data, "fixtime")
                    hist_df = transformation.extract_value_from_json_column(
                        subject_prev_data,
                        "extra__observation_details",
                        ["battery", "mainVoltage", "batt", "power"],
                        FieldType.FLOAT,
                        "voltage",
                    )
                    logger.info(f"Using previous period data for {subject_name} baseline ({len(hist_df)} records)")
                else:
                    hist_df = pd.DataFrame()
                    logger.warning(f"No previous period data found for {subject_name}")
            else:
                # Use historical data from before time_range.since (original behavior)
                hist_df = transformation.filter_df(dataframe, "fixtime", ComparisonOperator.LT, time_range.since)
                logger.info(f"Using historical data before {time_range.since} for {subject_name} baseline")

            if curr_df.empty and hist_df.empty:
                logger.warning(f"Skipping {subject_name}: no data in current or historic range")
                skipped_count += 1
                continue
                
            if hist_df.empty:
                print(f"WARNING: No historical data for {subject_name}")
                print(f"FALLBACK: Using current period data for baseline calculations")
                hist_df = curr_df.copy()
                
            volt_upper = dataframe_column_percentile(hist_df, "voltage", 97.5)
            volt_lower = dataframe_column_percentile(hist_df, "voltage", 2.5)
            volt_mean = dataframe_column_mean(hist_df, "voltage")

            if volt_upper == volt_lower:
                volt_upper_diff = apply_arithmetic_operation(volt_upper, 0.025, "multiply")
                volt_upper = apply_arithmetic_operation(volt_upper, volt_upper_diff, "add")
                volt_lower_diff = apply_arithmetic_operation(volt_lower, 0.025, "multiply")
                volt_lower = apply_arithmetic_operation(volt_lower, volt_lower_diff, "subtract")

            transformation.assign_value(curr_df, "max", volt_upper)
            transformation.assign_value(curr_df, "min", volt_lower)
            transformation.assign_value(curr_df, "mean", volt_mean)

            hist_lower_y = dataframe_column_min(hist_df, "voltage")
            curr_lower_y = dataframe_column_min(curr_df, "voltage")
            lower_y = apply_arithmetic_operation(hist_lower_y, curr_lower_y, "min")
            lower_y_diff = apply_arithmetic_operation(lower_y, 0.1, "multiply")
            lower_y = apply_arithmetic_operation(lower_y, lower_y_diff, "subtract")

            hist_upper_y = dataframe_column_max(hist_df, "voltage")
            curr_upper_y = dataframe_column_max(curr_df, "voltage")
            upper_y = apply_arithmetic_operation(hist_upper_y, curr_upper_y, "max")
            upper_y_diff = apply_arithmetic_operation(upper_y, 0.1, "multiply")

            html_output = draw_historic_timeseries(
                dataframe=curr_df,
                current_value_column="voltage",
                current_value_title="Current Voltage",
                historic_min_column="min",
                historic_max_column="max",
                historic_band_title="Historic 2.5% - 97.5%",
                historic_mean_column="mean",
                time_column="fixtime",
                upper_lower_band_style=PlotStyle(
                    mode="lines",
                    line=LineStyle(color="rgba(255,255,255,0)"),
                    fillcolor="rgba(0,176,246,0.2)"
                ),
                historic_mean_style=PlotStyle(
                    mode="lines",
                    line=LineStyle(color="Red", dash="dot")
                ),
                layout_style=LayoutStyle(
                    yaxis=AxisStyle(range=[lower_y, upper_y], title="Collar Voltage"),
                    xaxis=AxisStyle(title="Time"),
                ),
            )

            safe_name = str(subject_name).replace(" ", "_").replace("/", "_")
            file_path = persist_text(html_output, str(output_dir), f"{safe_name}.html")
            file_paths.append(file_path)
            processed_count += 1

            logger.info(f"Saved chart for {subject_name} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {subject_name}: {e}", exc_info=True)
            skipped_count += 1
            continue

    # Log summary
    logger.info(f"Processing complete: {processed_count} charts generated, {skipped_count} subjects skipped")
    print(f"Processing complete: {processed_count} charts generated, {skipped_count} subjects skipped")
    
    # CRITICAL: Return the file paths
    return file_paths
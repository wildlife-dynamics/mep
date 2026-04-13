import os
import pandas as pd
from pydantic import Field
from typing import Annotated, List
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.io import persist_text
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.tasks import transformation
from ecoscope_workflows_core.tasks.analysis._aggregation import (
    dataframe_column_first_unique,
    dataframe_column_percentile,
    dataframe_column_mean,
    apply_arithmetic_operation,
    dataframe_column_min,
    dataframe_column_max,
)
from ecoscope_workflows_core.tasks.transformation._extract import FieldType
from ecoscope_workflows_core.tasks.transformation._filter import ComparisonOperator
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import (
    AxisStyle,
    LayoutStyle,
    LineStyle,
    PlotStyle,
    draw_historic_timeseries,
)


@task
def extract_voltage_columns(
    relocs: Annotated[
        AnyGeoDataFrame,
        Field(description="Relocation geodataframe to prepare for voltage analysis", exclude=True),
    ],
) -> AnyGeoDataFrame:
    """Extract and type the assigned_range columns needed for voltage chart generation."""
    relocs = transformation.extract_column_as_type(
        relocs,
        "extra__subjectsource__assigned_range",
        FieldType.SERIES,
        "extra.extra.subjectsource__assigned_range.",
    )
    relocs = transformation.extract_column_as_type(
        relocs,
        "extra.extra.subjectsource__assigned_range.upper",
        FieldType.DATETIME,
        "extra.extra.subjectsource__assigned_range.upper",
    )
    return relocs


@task
def generate_subject_voltage_chart(
    subject_df: Annotated[
        AnyGeoDataFrame,
        Field(description="Single subject's relocation data with assigned_range columns extracted", exclude=True),
    ],
    subject_name: Annotated[str, Field(description="Subject name used for the chart title")],
    time_range: Annotated[TimeRange, Field(description="Analysis time range")],
    previous_subject_df: Annotated[
        AnyGeoDataFrame | None,
        Field(default=None, description="Optional previous period data for the same subject", exclude=True),
    ] = None,
) -> str:
    """
    For a single subject, extract voltage, compute historic band statistics, and return an HTML chart string.
    Falls back to using current period data as the baseline if no historical data is available.
    """
    subject_df = transformation.sort_values(subject_df, "fixtime")
    subject_df = transformation.extract_value_from_json_column(
        subject_df,
        "extra__observation_details",
        ["battery", "mainVoltage", "batt", "power"],
        FieldType.FLOAT,
        "voltage",
    )

    curr_df = transformation.filter_df(subject_df, "fixtime", ComparisonOperator.GE, time_range.since)
    curr_df = transformation.filter_df(curr_df, "fixtime", ComparisonOperator.LT, time_range.until)

    if previous_subject_df is not None and not previous_subject_df.empty:
        hist_df = transformation.sort_values(previous_subject_df, "fixtime")
        hist_df = transformation.extract_value_from_json_column(
            hist_df,
            "extra__observation_details",
            ["battery", "mainVoltage", "batt", "power"],
            FieldType.FLOAT,
            "voltage",
        )
    else:
        hist_df = transformation.filter_df(subject_df, "fixtime", ComparisonOperator.LT, time_range.since)

    if hist_df.empty:
        hist_df = curr_df.copy()

    volt_upper = dataframe_column_percentile(hist_df, "voltage", 97.5)
    volt_lower = dataframe_column_percentile(hist_df, "voltage", 2.5)
    volt_mean = dataframe_column_mean(hist_df, "voltage")

    if volt_upper == volt_lower:
        upper_diff = apply_arithmetic_operation(volt_upper, 0.025, "multiply")
        volt_upper = apply_arithmetic_operation(volt_upper, upper_diff, "add")
        lower_diff = apply_arithmetic_operation(volt_lower, 0.025, "multiply")
        volt_lower = apply_arithmetic_operation(volt_lower, lower_diff, "subtract")

    hist_lower_y = dataframe_column_min(hist_df, "voltage")
    curr_lower_y = dataframe_column_min(curr_df, "voltage")
    lower_y = apply_arithmetic_operation(hist_lower_y, curr_lower_y, "min")
    lower_y_diff = apply_arithmetic_operation(lower_y, 0.1, "multiply")
    lower_y = apply_arithmetic_operation(lower_y, lower_y_diff, "subtract")

    hist_upper_y = dataframe_column_max(hist_df, "voltage")
    curr_upper_y = dataframe_column_max(curr_df, "voltage")
    upper_y = apply_arithmetic_operation(hist_upper_y, curr_upper_y, "max")

    transformation.assign_value(curr_df, "max", volt_upper)
    transformation.assign_value(curr_df, "min", volt_lower)
    transformation.assign_value(curr_df, "mean", volt_mean)

    return draw_historic_timeseries(
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
            fillcolor="rgba(0,176,246,0.2)",
        ),
        historic_mean_style=PlotStyle(mode="lines", line=LineStyle(color="Red", dash="dot")),
        layout_style=LayoutStyle(
            yaxis=AxisStyle(range=[lower_y, upper_y], title="Collar Voltage"),
            xaxis=AxisStyle(title="Time"),
            title=f"{subject_name}",
            title_x=0.5,
            font_size=14,
            font_color="#222222",
            plot_bgcolor="#f5f5f5",
            hovermode="x unified",
        ),
    )


@task
def process_collar_voltage_charts(
    relocs: Annotated[
        AnyGeoDataFrame,
        Field(description="The relocation geodataframe for current period.", exclude=True),
    ],
    time_range: Annotated[TimeRange, Field(description="Time range for voltage analysis")],
    output_dir: Annotated[str, Field(description="Directory to save output HTML charts")],
    previous_relocs: Annotated[
        AnyGeoDataFrame | None,
        Field(
            default=None,
            description="Optional relocation geodataframe.",
            exclude=True,
        ),
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

    relocs = extract_voltage_columns(relocs)
    if previous_relocs is not None:
        previous_relocs = extract_voltage_columns(previous_relocs)

    groups = relocs.groupby(by=["extra__subject__name", "extra__subjectsource__id"])
    print(f"Processing voltage charts for {len(groups)} subjects")

    file_paths = []
    skipped_count = 0

    for (subject_name, subjectsource_id), subject_df in groups:
        try:
            upperbound = dataframe_column_first_unique(subject_df, "extra.extra.subjectsource__assigned_range.upper")
            if not pd.isna(upperbound) and upperbound < pd.to_datetime(time_range.since):
                print(f"Skipping {subject_name}: collar deactivated before time range")
                skipped_count += 1
                continue

            previous_subject_df = None
            if previous_relocs is not None:
                mask = (previous_relocs["extra__subject__name"] == subject_name) & (
                    previous_relocs["extra__subjectsource__id"] == subjectsource_id
                )
                filtered = previous_relocs[mask].copy()
                previous_subject_df = filtered if not filtered.empty else None

            html_output = generate_subject_voltage_chart(subject_df, subject_name, time_range, previous_subject_df)
            safe_name = str(subject_name).replace(" ", "_").replace("/", "_")
            file_path = persist_text(html_output, str(output_dir), f"{safe_name}.html")
            file_paths.append(file_path)
            print(f"Saved chart for {subject_name} to {file_path}")

        except Exception as e:
            print(f"Failed to process {subject_name}: {e}")
            skipped_count += 1
            continue

    print(f"Processing complete: {len(file_paths)} charts generated, {skipped_count} subjects skipped")
    return file_paths

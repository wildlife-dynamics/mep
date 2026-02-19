import pandas as pd
from pydantic import Field
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.tasks import results
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.tasks import analysis, transformation
from ecoscope_workflows_core.tasks.transformation._extract import FieldType
from ecoscope_workflows_core.tasks.transformation._filter import ComparisonOperator
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import AxisStyle, LayoutStyle, LineStyle, PlotStyle


@task
def calculate_collar_voltage(
    relocs: Annotated[AnyGeoDataFrame, Field(description="The relocation geodataframe.", exclude=True)],
    time_range: Annotated[TimeRange, Field(description="Time range filter")],
) -> str:
    relocs = transformation.extract_column_as_type(
        relocs, "extra__subjectsource__assigned_range", FieldType.SERIES, "extra.extra.subjectsource__assigned_range."
    )
    relocs = transformation.extract_column_as_type(
        relocs,
        "extra.extra.subjectsource__assigned_range.upper",
        FieldType.DATETIME,
        "extra.extra.subjectsource__assigned_range.upper",
    )

    groups = relocs.groupby(by=["extra__subject__name", "extra__subjectsource__id"])

    for _, dataframe in groups:
        subjectsource_upperbound = analysis.dataframe_column_first_unique(
            dataframe, "extra.extra.subjectsource__assigned_range.upper"
        )

        if not pd.isna(subjectsource_upperbound) and subjectsource_upperbound < pd.to_datetime(time_range.since):
            continue

        dataframe = transformation.sort_values(dataframe, "fixtime")
        dataframe = transformation.extract_value_from_json_column(
            dataframe,
            "extra__observation_details",
            [
                "battery",
                "mainVoltage",
                "batt",
                "power",
            ],
            FieldType.FLOAT,
            "voltage",
        )

        curr_df = transformation.filter_df(
            dataframe, column_name="fixtime", op=ComparisonOperator.GE, value=time_range.since
        )
        curr_df = transformation.filter_df(
            curr_df, column_name="fixtime", op=ComparisonOperator.LT, value=time_range.until
        )
        hist_df = transformation.filter_df(
            dataframe, column_name="fixtime", op=ComparisonOperator.LT, value=time_range.since
        )

        if curr_df.empty and hist_df.empty:
            continue

        volt_upper = analysis.dataframe_column_percentile(hist_df, "voltage", 97.5)
        volt_lower = analysis.dataframe_column_percentile(hist_df, "voltage", 2.5)
        volt_mean = analysis.dataframe_column_mean(hist_df, "voltage")

        if volt_upper == volt_lower:
            volt_upper_diff = analysis.apply_arithmetic_operation(volt_upper, 0.025, "multiply")
            volt_upper = analysis.apply_arithmetic_operation(volt_upper, volt_upper_diff, "add")
            volt_lower_diff = analysis.apply_arithmetic_operation(volt_lower, 0.025, "multiply")
            volt_lower = analysis.apply_arithmetic_operation(volt_lower, volt_lower_diff, "subtract")

        transformation.assign_value(curr_df, "max", volt_upper)
        transformation.assign_value(curr_df, "min", volt_lower)
        transformation.assign_value(curr_df, "mean", volt_mean)

        hist_lower_y = analysis.dataframe_column_min(hist_df, "voltage")
        curr_lower_y = analysis.dataframe_column_min(curr_df, "voltage")
        lower_y = analysis.apply_arithmetic_operation(hist_lower_y, curr_lower_y, "min")
        lower_y_diff = analysis.apply_arithmetic_operation(lower_y, 0.1, "multiply")
        lower_y = analysis.apply_arithmetic_operation(lower_y, lower_y_diff, "subtract")

        hist_upper_y = analysis.dataframe_column_max(hist_df, "voltage")
        curr_upper_y = analysis.dataframe_column_max(curr_df, "voltage")
        upper_y = analysis.apply_arithmetic_operation(hist_upper_y, curr_upper_y, "max")
        upper_y_diff = analysis.apply_arithmetic_operation(upper_y, 0.1, "multiply")
        upper_y = analysis.apply_arithmetic_operation(upper_y, upper_y_diff, "add")

        html_output = results.draw_historic_timeseries(
            dataframe=curr_df,
            current_value_column="voltage",
            current_value_title="Current Voltage",
            historic_min_column="min",
            historic_max_column="max",
            historic_band_title="Historic 2.5% - 97.5%",
            historic_mean_column="mean",
            time_column="fixtime",
            upper_lower_band_style=PlotStyle(
                mode="lines", line=LineStyle(color="rgba(255,255,255,0)"), fillcolor="rgba(0,176,246,0.2)"
            ),
            historic_mean_style=PlotStyle(mode="lines", line=LineStyle(color="Red", dash="dot")),
            layout_style=LayoutStyle(
                yaxis=AxisStyle(range=[lower_y, upper_y], title="Collar Voltage"), xaxis=AxisStyle(title="Time")
            ),
        )

        return html_output

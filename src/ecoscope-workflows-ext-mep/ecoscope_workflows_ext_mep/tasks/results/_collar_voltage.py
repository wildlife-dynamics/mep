from wt_registry import register
from typing import Annotated, List
from pydantic import Field
from wt_task.skip import SkippedDependencyFallback
from ecoscope.platform.tasks import transformation
from ecoscope.platform.annotations import AnyGeoDataFrame, EmptyDataFrame
from ecoscope.platform.tasks.skip import skip_gdf_fallback_to_none
from ecoscope.platform.tasks.analysis._aggregation import (
    dataframe_column_percentile,
    dataframe_column_mean,
    apply_arithmetic_operation,
    dataframe_column_min,
    dataframe_column_max,
)
from ecoscope.platform.tasks.results._ecoplot import (
    AxisStyle,
    LayoutStyle,
    LineStyle,
    PlotStyle,
    draw_historic_timeseries,
)

@register()
def plot_historic_voltage(
    current_relocs: AnyGeoDataFrame,
    previous_relocs: Annotated[
        AnyGeoDataFrame | EmptyDataFrame | None,
        Field(description="Relocations from prior runs, used to build the historic voltage band."),
        SkippedDependencyFallback(skip_gdf_fallback_to_none),
    ],
    column: str = "voltage",
)->str:
    """Plot current collar voltage against a historic min/max/mean band.

    Args:
        current_relocs: Relocations from the current run; provides the plotted voltage
            series and, when no history is available, doubles as the historic band.
        previous_relocs: Relocations from prior runs, used to compute the historic
            min/max/mean band. May arrive as `None` (no prior run), an empty dataframe,
            or a `SkipSentinel` (normalized to `None` by `SkippedDependencyFallback`)
            when the upstream historic-relocations task was skipped. In all of those
            cases we fall back to `current_relocs`, so the band collapses to the
            current value rather than the task failing.
        column: Name of the voltage column to plot.

    Returns:
        HTML string of the rendered historic voltage timeseries plot.
    """
    if previous_relocs is None or previous_relocs.empty:
        previous_relocs = current_relocs.copy()

    volt_upper = dataframe_column_percentile(previous_relocs, column, 97.5)
    volt_lower = dataframe_column_percentile(previous_relocs, column, 2.5)
    volt_mean = dataframe_column_mean(previous_relocs, column)

    # Widen the band when upper and lower collapse onto the same value
    if volt_upper == volt_lower:
        upper_diff = apply_arithmetic_operation(volt_upper, 0.025, "multiply")
        volt_upper = apply_arithmetic_operation(volt_upper, upper_diff, "add")
        lower_diff = apply_arithmetic_operation(volt_lower, 0.025, "multiply")
        volt_lower = apply_arithmetic_operation(volt_lower, lower_diff, "subtract")

    # y-axis lower bound: min across history + current, padded 15%
    lower_y = apply_arithmetic_operation(
        dataframe_column_min(previous_relocs, column),
        dataframe_column_min(current_relocs, column),
        "min",
    )
    lower_y = apply_arithmetic_operation(
        lower_y, apply_arithmetic_operation(lower_y, 0.15, "multiply"), "subtract"
    )

    # y-axis upper bound: max across history + current, padded 15%
    upper_y = apply_arithmetic_operation(
        dataframe_column_max(previous_relocs, column),
        dataframe_column_max(current_relocs, column),
        "max",
    )
    upper_y = apply_arithmetic_operation(
        upper_y, apply_arithmetic_operation(upper_y, 0.15, "multiply"), "add"
    )

    transformation.assign_value(current_relocs, "max", volt_upper)
    transformation.assign_value(current_relocs, "min", volt_lower)
    transformation.assign_value(current_relocs, "mean", volt_mean)

    return draw_historic_timeseries(
        dataframe=current_relocs,
        current_value_column=column,
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
            title=None,
            title_x=0.5,
            font_size=14,
            font_color="#222222",
            plot_bgcolor="#f5f5f5",
            hovermode="x unified",
        ),
    )
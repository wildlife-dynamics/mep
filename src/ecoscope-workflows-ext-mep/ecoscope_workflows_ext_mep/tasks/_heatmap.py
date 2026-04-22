from ecoscope_workflows_ext_big_life.tasks._chart import TimeFrequency
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import ExportArgs, LayoutStyle
import pandas as pd
from ecoscope_workflows_core.annotations import (
    AdvancedField,
    DataFrame,
    JsonSerializableDataFrameModel,
)

from pydantic import Field
import plotly.graph_objects as go
from typing import Optional, List, Annotated
from pydantic.json_schema import SkipJsonSchema
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.tasks.analysis._summary import AggOperations


def _resolve_category_order(categories, group_order=None, ascending: bool = True) -> list:
    """Return a deterministic ordered list from *categories*."""
    if group_order is not None:
        ordered = [c for c in group_order if c in categories]
        ordered.extend(sorted(c for c in categories if c not in ordered))
        return ordered
    return sorted(categories, reverse=not ascending)


def _build_layout_kwargs(layout_style: LayoutStyle) -> dict:
    raw = layout_style.model_dump(exclude_none=True)
    layout_kws = {}

    # Font fields get nested under "font"
    font = {}
    for key in ("font_size", "font_color", "font_style"):
        if key in raw:
            font[key.replace("font_", "")] = raw.pop(key)
    if font:
        layout_kws["font"] = font

    # Legend title gets nested
    if "legend_title" in raw:
        layout_kws["legend"] = {"title": raw.pop("legend_title")}

    # xaxis / yaxis are AxisStyle objects — dump them too
    for axis in ("xaxis", "yaxis"):
        if axis in raw:
            layout_kws[axis] = raw.pop(axis)  # already a dict from model_dump

    layout_kws.update(raw)  # remaining flat fields (title, title_x, title_y, etc.)
    return layout_kws


def custom_heatmap(
    dataframe: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    value_column: str,
    agg_function: str,
    time_frequency: Optional[str] = None,
    time_column: Optional[str] = None,
    time_axis: str = "x",
    normalize: bool = False,
    colorscale: str = "YlOrRd",
    show_values: bool = True,
    x_order: Optional[List[str]] = None,
    y_order: Optional[List[str]] = None,
    value_label: Optional[str] = None,
    layout_kwargs: dict | None = None,
) -> go.Figure:
    """
    Build a heatmap where *y_axis* categories form rows, *x_axis* categories form
    columns, and each cell shows *value_column* aggregated by *agg_function*.

    When *time_frequency* and *time_column* are provided, the axis indicated by
    *time_axis* ('x' or 'y') is replaced by resampled time buckets instead of a
    categorical column.

    Parameters
    ----------
    dataframe      : Source records.
    x_axis         : Column supplying column labels (heatmap x). Ignored when
                     time_axis='x' and time_frequency is set.
    y_axis         : Column supplying row labels (heatmap y). Ignored when
                     time_axis='y' and time_frequency is set.
    value_column   : Numeric column aggregated per cell.
    agg_function   : Aggregation string: "sum", "mean", "count", etc.
    time_frequency : Pandas offset alias, e.g. "ME", "QE", "YE". When None,
                     behaves as a purely categorical heatmap.
    time_column    : Datetime column to resample on. Required if time_frequency
                     is set.
    time_axis      : Which axis gets time bins: 'x' (default) or 'y'.
    normalize      : If True, row-normalise each row to % of its total.
    colorscale     : Any Plotly colorscale name (e.g. "YlOrRd", "Viridis").
    show_values    : If True, print aggregated values inside each cell.
    x_order        : Explicit left-to-right order for x categories. Ignored if
                     time_axis='x' (time buckets are naturally ordered).
    y_order        : Explicit top-to-bottom order for y categories. Ignored if
                     time_axis='y'.
    value_label    : Label for colorbar / hover. Defaults based on *normalize*.
    layout_kwargs  : Extra kwargs forwarded to fig.update_layout().
    """
    df = dataframe.copy()
    if time_frequency is not None:
        if time_column is None:
            raise ValueError("time_column must be provided when time_frequency is set.")
        if time_axis not in ("x", "y"):
            raise ValueError("time_axis must be 'x' or 'y'.")

        df[time_column] = pd.to_datetime(df[time_column], utc=True).dt.tz_localize(None)

        # Resample the time column into period buckets, keyed on the opposite axis
        group_col = y_axis if time_axis == "x" else x_axis

        frames = []
        for category, cat_df in df.groupby(group_col):
            rs = (
                cat_df.set_index(time_column)
                .sort_index()[value_column]
                .resample(time_frequency)
                .agg(agg_function)
                .reset_index()
            )
            rs[group_col] = category
            frames.append(rs)

        agg_df = pd.concat(frames, ignore_index=True).fillna(0)

        # Human-readable period labels (Jan 2024, Q1 2024, 2024, etc.)
        agg_df["_period_label"] = _format_period_labels(agg_df[time_column], time_frequency)

        period_col = "_period_label"
        if time_axis == "x":
            matrix = agg_df.pivot_table(
                index=group_col,
                columns=period_col,
                values=value_column,
                aggfunc="sum",
                fill_value=0,
            )
            # Preserve chronological order on x
            period_order = agg_df.sort_values(time_column)[period_col].drop_duplicates().tolist()
            matrix = matrix.reindex(columns=period_order, fill_value=0)
            x_order = period_order  # time order wins
        else:
            matrix = agg_df.pivot_table(
                index=period_col,
                columns=group_col,
                values=value_column,
                aggfunc="sum",
                fill_value=0,
            )
            period_order = agg_df.sort_values(time_column)[period_col].drop_duplicates().tolist()
            matrix = matrix.reindex(index=period_order, fill_value=0)
            y_order = period_order

    else:
        matrix = df.groupby([y_axis, x_axis])[value_column].agg(agg_function).unstack(fill_value=0)
    if normalize:
        row_totals = matrix.sum(axis=1).replace(0, pd.NA)
        matrix = matrix.div(row_totals, axis=0).mul(100).fillna(0).round(1)
    y_categories = matrix.index.tolist()
    x_categories = matrix.columns.tolist()
    ordered_y = _resolve_category_order(y_categories, y_order, ascending=True)

    if time_frequency is not None and time_axis == "x":
        ordered_x = x_order  # already chronological
    elif x_order is not None:
        ordered_x = _resolve_category_order(x_categories, x_order, ascending=True)
    else:
        ordered_x = matrix.sum(axis=0).sort_values(ascending=False).index.tolist()

    matrix = matrix.reindex(index=ordered_y, columns=ordered_x, fill_value=0)

    z = matrix.values
    resolved_label = value_label or ("Share (%)" if normalize else agg_function.title())
    hover_fmt = "%{z:.1f}%" if normalize else "%{z:.0f}"
    text_fmt = "%{text:.1f}" if normalize else "%{text:.0f}"

    x_axis_label = time_column if (time_frequency and time_axis == "x") else x_axis
    y_axis_label = time_column if (time_frequency and time_axis == "y") else y_axis

    heatmap_kwargs = dict(
        z=z,
        x=list(matrix.columns),
        y=list(matrix.index),
        colorscale=colorscale,
        colorbar=dict(title=resolved_label),
        hovertemplate=(
            f"{y_axis_label}: %{{y}}<br>" f"{x_axis_label}: %{{x}}<br>" f"{resolved_label}: {hover_fmt}<extra></extra>"
        ),
    )
    if show_values:
        heatmap_kwargs["text"] = z
        heatmap_kwargs["texttemplate"] = text_fmt
        heatmap_kwargs["textfont"] = dict(size=11)

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs), layout=layout_kwargs)

    fig.update_layout(
        xaxis=dict(title=x_axis_label, tickangle=-40, type="category"),
        yaxis=dict(title=y_axis_label, autorange="reversed", type="category"),
    )
    return fig


def _format_period_labels(series: pd.Series, freq: str) -> pd.Series:
    """Turn resampled timestamps into readable labels based on freq."""
    freq_base = freq.split("-")[0].upper()
    if freq_base in ("YE", "YS", "Y", "A", "AS", "AE"):
        return series.dt.strftime("%Y")
    if freq_base in ("QE", "QS", "Q"):
        return series.dt.to_period("Q").astype(str)  # e.g. '2024Q1'
    if freq_base in ("ME", "MS", "M", "2ME", "2MS", "6ME", "6MS"):
        return series.dt.strftime("%b %Y")
    if freq_base == "W":
        return series.dt.strftime("%d %b %Y")
    return series.dt.strftime("%d %b %Y")  # daily / fallback


@task
def draw_custom_heatmap(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    x_axis: Annotated[
        str,
        Field(description="The dataframe column whose values form heatmap columns (x)."),
    ],
    y_axis: Annotated[
        str,
        Field(description="The dataframe column whose values form heatmap rows (y)."),
    ],
    value_column: Annotated[
        str,
        Field(description="The numeric dataframe column to aggregate per (y, x) cell."),
    ],
    agg_function: Annotated[
        AggOperations,
        Field(description="The aggregate function to apply per cell."),
    ],
    time_frequency: Annotated[
        TimeFrequency | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description=(
                "If set, the heatmap replaces one axis with time buckets resampled "
                "at this frequency. Requires time_column."
            ),
        ),
    ] = None,
    time_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Datetime column to resample on. Required if time_frequency is set.",
        ),
    ] = None,
    time_axis: Annotated[
        str,
        AdvancedField(
            default="x",
            description="Which axis becomes time: 'x' (default) or 'y'.",
        ),
    ] = "x",
    normalize: Annotated[
        bool,
        Field(
            default=False,
            description="If True, each row is normalised to % of its row total.",
        ),
    ] = False,
    colorscale: Annotated[
        str,
        Field(
            default="YlOrRd",
            description="Plotly colorscale name (e.g. 'YlOrRd', 'Viridis', 'Blues').",
        ),
    ] = "YlOrRd",
    show_values: Annotated[
        bool,
        Field(
            default=True,
            description="If True, aggregated values are printed inside each cell.",
        ),
    ] = True,
    x_order: Annotated[
        List[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description=(
                "Custom left-to-right order for x categories. Ignored when the x "
                "axis represents time (chronological order is enforced)."
            ),
        ),
    ] = None,
    y_order: Annotated[
        List[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description=("Custom top-to-bottom order for y categories. Ignored when the y " "axis represents time."),
        ),
    ] = None,
    value_label: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Label for the colorbar and hover readout. Defaults based on normalize.",
        ),
    ] = None,
    layout_style: Annotated[
        LayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Layout kwargs passed to plotly.go.Figure(layout).",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description=(
                "The id of the dashboard widget that this tile layer belongs to. "
                "If set this MUST match the widget title as defined downstream in "
                "create_widget tasks."
            ),
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    """
    Generates a heatmap of *value_column* aggregated across two dimensions.

    Without *time_frequency*, this is a pure categorical heatmap (y × x).
    With *time_frequency* and *time_column* set, one axis (controlled by
    *time_axis*) is replaced with resampled time buckets — e.g. ranch × month,
    species × quarter, etc.
    """
    layout_kws = _build_layout_kwargs(layout_style) if layout_style else {}

    freq_str = time_frequency.to_pandas_freq() if time_frequency is not None else None

    fig = custom_heatmap(
        dataframe=pd.DataFrame(dataframe),
        x_axis=x_axis,
        y_axis=y_axis,
        value_column=value_column,
        agg_function=agg_function,
        time_frequency=freq_str,
        time_column=time_column,
        time_axis=time_axis,
        normalize=normalize,
        colorscale=colorscale,
        show_values=show_values,
        x_order=x_order,
        y_order=y_order,
        value_label=value_label,
        layout_kwargs=layout_kws or None,
    )

    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

import logging
import ecoscope
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import Field
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from ecoscope.plotting.plot import nsd
from plotly.subplots import make_subplots
from ecoscope.trajectory import Trajectory
from ecoscope.relocations import Relocations
from typing import Union, Annotated, Optional
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import ExportArgs
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

logger = logging.getLogger(__name__)


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
    return fig


def _load_seasons_df(seasons_df: Union[str, Path, AnyDataFrame]) -> AnyDataFrame:
    if isinstance(seasons_df, pd.DataFrame):
        return seasons_df
    normalized_path = remove_file_scheme(str(seasons_df))
    p = Path(normalized_path)
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    elif p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    else:
        return pd.read_csv(p)


@task
def draw_season_nsd_plot(
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
        raise ValueError("Relocations gdf is empty.")

    seasons_df = _load_seasons_df(seasons_df)
    relocations_gdf = Relocations.from_gdf(relocations_gdf)
    figure = nsd(relocations_gdf)
    figure = add_seasons_square(figure, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@task
def draw_season_speed_plot(
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
        raise ValueError("Relocations gdf is empty.")

    seasons_df = _load_seasons_df(seasons_df)
    relocations_gdf = Relocations.from_gdf(relocations_gdf)
    trajs_gdf = Trajectory.from_relocations(relocations_gdf)
    figure = ecoscope.plotting.speed(trajs_gdf)
    figure = add_seasons_square(figure, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@task
def draw_season_mcp_plot(
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
        raise ValueError("Relocations gdf is empty.")

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

    if geodataframe is None or geodataframe.empty:
        raise ValueError("Relocations gdf is empty.")

    required_cols = ["fixtime"]
    missing_cols = [col for col in required_cols if col not in geodataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    geodataframe = geodataframe.dropna(subset=["fixtime"]).copy()

    ys = [0]
    if collar_events is not None and not collar_events.empty:
        collar_events = collar_events.dropna(subset=["time"]).copy()

        if not collar_events.empty:
            times = collar_events["time"].to_list()
            times.append(geodataframe["fixtime"].iloc[-1])

            xs = [[times[i]] * 3 + [times[i + 1]] for i in range(len(collar_events))]
            ys = [[0, i + 1, 0, 0] for i in range(len(collar_events))]
            colors = collar_events["priority_label"]

            for x, y, color in zip(xs, ys, colors):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        line_color=color,
                        mode="lines",
                        showlegend=False,
                    )
                )

            fig.update_layout(
                annotations=[
                    go.layout.Annotation(
                        x=row.time,
                        y=i,
                        text=f"{row.event_type}<br>{row.time.date()}",
                        showarrow=False,
                    )
                    for i, (_, row) in enumerate(collar_events.iterrows(), 1)
                ]
            )

    x = geodataframe["fixtime"]
    max_y = max([max(y_list) for y_list in ys]) if isinstance(ys[0], list) else max(ys)
    y = np.full(len(x), max_y / 10 if max_y > 0 else 0.5)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line_color="rgb(0,0,255)",
            mode="markers",
            marker=dict(size=4, color="rgb(0, 100, 200)"),
            showlegend=False,
            hovertemplate="Relocation: %{x}",
        )
    )
    fig.update_layout(
        margin_l=50,
        margin_r=10,
        margin_t=25,
        margin_b=15,
        yaxis_visible=False,
        showlegend=False,
    )
    return fig


@task
def draw_season_collared_plot(
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
        raise ValueError("Relocations gdf is empty.")

    subject_name = relocations_gdf["subject_name"].unique()[0]

    if events_gdf is None or events_gdf.empty:
        logger.warning(f"No events data for subject '{subject_name}'.")
        events_gdf = None
    elif filter_col not in events_gdf.columns:
        raise ValueError(f"Column '{filter_col}' not found. Available: {', '.join(events_gdf.columns)}")
    else:
        events_gdf = events_gdf[events_gdf[filter_col] == subject_name]
        if events_gdf.empty:
            logger.warning(f"No events found for subject '{subject_name}'.")
            events_gdf = None
    fig = collar_event_timeline_plot(relocations_gdf, events_gdf)
    figure = add_seasons_square(fig, seasons_df)
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

import pandas as pd 
import numpy as np
import ecoscope
import plotly.graph_objs as go  # type: ignore[import-untyped]
from wt_registry import register
from ecoscope.trajectory import Trajectory
from ecoscope.relocations import Relocations
from typing import Union, Annotated, Optional,Field
from ecoscope.platform.schemas import RelocationsGDFSchema
from ecoscope.platform.tasks.results._ecoplot import ExportArgs
from ecoscope.platform.annotations import AnyGeoDataFrame, AnyDataFrame


DEFAULT_SEASON_COLORS = {
    "wet": "rgba(30, 144, 255, 0.18)",   # blue
    "dry": "rgba(30, 144, 255, 0.05)",   # amber
}
FALLBACK_COLOR = "rgba(120, 120, 120, 0.10)"


def nsd(
    relocations: RelocationsGDFSchema,
    legend_column: str | None = None,
) -> go.Figure:
    """Net Squared Displacement (km²) over time, one trace per subject.

    Parameters
    ----------
    legend_column: Column to use for trace names in the legend (e.g. 'subject_name').
        Falls back to 'groupby_col' if None or not present.
    """
    if relocations.empty:
        fig = go.Figure()
        fig.update_layout(
            yaxis_title="NSD (km²)",
            annotations=[dict(text="No relocation data", showarrow=False)],
        )
        return fig

    # Resolve which column labels the traces
    if legend_column is not None and legend_column not in relocations.columns:
        print(f"legend_column '{legend_column}' not found; using 'groupby_col'. "
              f"Available: {list(relocations.columns)}")
        legend_column = None
    label_col = legend_column or "groupby_col"

    # Work on a copy in a projected CRS — never mutate the caller's frame.
    gdf = relocations.to_crs(relocations.estimate_utm_crs())

    fig = go.Figure()

    for subject, group in gdf.groupby("groupby_col"):
        group = group.sort_values("fixtime")
        origin = group.geometry.iloc[0]
        nsd_km2 = group.distance(origin) ** 2 / 1_000_000  # m² -> km²

        # Label: first non-null value of label_col in this group, else the group key
        label = group[label_col].dropna().iloc[0] if group[label_col].notna().any() else subject

        fig.add_trace(
            go.Scatter(
                x=group["fixtime"],
                y=nsd_km2,
                mode="lines",
                name=str(label),
                hovertemplate="%{x|%d %b %Y %H:%M}<br>NSD: %{y:.1f} km²<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        margin=dict(b=15, l=50, r=10, t=25),
        yaxis_title="NSD (km²)",
        showlegend=gdf["groupby_col"].nunique() > 1,
        font=dict(size=12, color="#222222"),
        plot_bgcolor="#f5f5f5",
    )
    fig.update_xaxes(range=[gdf["fixtime"].min(), gdf["fixtime"].max()])
    return fig

def add_seasons_square(
    fig: go.Figure,
    season_df: AnyDataFrame,
    colors: dict[str, str] | None = None,
    show_labels: bool = False,
    show_legend: bool = True,
) -> go.Figure:
    """
    Shade seasonal intervals on a Plotly time-series figure.

    Args:
        fig: Plotly figure (e.g. from nsd()).
        season_df: DataFrame with columns ['start', 'end', 'season'].
        colors: Optional {season_name: rgba_color} override.
        show_labels: Annotate each band with its season name at the top.
        show_legend: Add one legend entry per season type.
    Returns:
        The figure, with season bands drawn below the data traces.
    """
    colors = {**DEFAULT_SEASON_COLORS, **(colors or {})}
    seen_seasons: set[str] = set()

    for idx, row in season_df.iterrows():
        try:
            start_dt = pd.to_datetime(row["start"])
            end_dt = pd.to_datetime(row["end"])
        except Exception as e:
            print(f"Skipping season row {idx}: unparseable dates ({e})")
            continue
        if pd.isna(start_dt) or pd.isna(end_dt) or end_dt <= start_dt:
            print(f"Skipping season row {idx}: invalid interval {row['start']!r} -> {row['end']!r}")
            continue

        season_type = str(row.get("season", "") or "").strip().lower()
        fillcolor = colors.get(season_type, FALLBACK_COLOR)

        fig.add_shape(
            type="rect",
            x0=start_dt, x1=end_dt,
            y0=0, y1=1, yref="paper",
            fillcolor=fillcolor,
            line_width=0,
            layer="below",
        )

        if show_labels:
            fig.add_annotation(
                x=start_dt + (end_dt - start_dt) / 2,
                y=0.5, yref="paper", yanchor="middle",
                text=season_type or "?",
                textangle=-90,
                showarrow=False,
                font=dict(size=8, color="gray"),
            )

        # One legend entry per season type, via invisible marker traces
        if show_legend and season_type and season_type not in seen_seasons:
            seen_seasons.add(season_type)
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=12, color=fillcolor, symbol="square"),
                    name=season_type.capitalize(),
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    return fig

@register()
def draw_season_nsd_plot(
    relocations_gdf: AnyGeoDataFrame,
    seasons_df: AnyDataFrame,
    legend_column: Union[str|None]=None,
    season_colors: Union[dict[str,str]|None]=None,
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
    #relocations_gdf = Relocations.from_gdf(relocations_gdf)
    figure = nsd(relocations=relocations_gdf,legend_column=legend_column)
    figure = add_seasons_square(
        fig =figure, 
        season_df = seasons_df,
        show_labels=False,
        colors=season_colors,
        show_legend=False
    )
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

def speed(trajectory: ecoscope.Trajectory) -> go.Figure:
    times = np.column_stack(
        [
            trajectory.gdf["segment_start"],
            trajectory.gdf["segment_start"],
            trajectory.gdf["segment_end"],
            trajectory.gdf["segment_end"],
        ]
    ).flatten()
    speeds = np.column_stack(
        [
            np.zeros(len(trajectory.gdf)),
            trajectory.gdf["speed_kmhr"],
            trajectory.gdf["speed_kmhr"],
            np.zeros(len(trajectory.gdf)),
        ]
    ).flatten()
    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(x=times, y=speeds))
    fig.update_layout(
        margin_b=15,
        margin_l=50,
        margin_r=10,
        margin_t=25,
        yaxis_title="Speed (km/h)",
        showlegend=False,
        font=dict(size=12, color="#222222"),
        plot_bgcolor="#f5f5f5",
    )
    return fig

@register()
def draw_season_speed_plot(
    relocations_gdf: AnyGeoDataFrame,
    seasons_df: AnyDataFrame,
    season_colors: Union[dict[str,str]|None]=None,
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
    x_min = relocations_gdf["fixtime"].min()
    x_max = relocations_gdf["fixtime"].max()
    relocations_gdf = Relocations.from_gdf(relocations_gdf)
    trajs_gdf = Trajectory.from_relocations(relocations_gdf)
    figure = speed(trajs_gdf)
    figure = add_seasons_square(
        fig =figure, 
        season_df = seasons_df,
        show_labels=False,
        colors=season_colors,
        show_legend=False
    )
    figure.update_xaxes(range=[x_min, x_max])
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


def mcp(relocations: ecoscope.Relocations) -> go.Figure:
    import shapely
    relocations.gdf.to_crs(relocations.gdf.estimate_utm_crs(), inplace=True)
    areas = []
    times = []
    total = shapely.geometry.GeometryCollection()
    for time, obs in relocations.gdf.groupby(pd.Grouper(key="fixtime", freq="1D"), as_index=False):
        if obs.size:
            total = total.union(obs.geometry.union_all()).convex_hull
            areas.append(total.area)
            times.append(time)
    areas_np = np.array(areas)
    times_np = np.array(times)
    times_np[0] = relocations.gdf["fixtime"].iat[0]
    times_np[-1] = relocations.gdf["fixtime"].iat[-1]
    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(x=times_np, y=areas_np / (1000**2)))
    fig.update_layout(
        margin_b=15,
        margin_l=50,
        margin_r=10,
        margin_t=25,
        yaxis_title="MCP Area (km^2)",
        showlegend=False,
        font=dict(size=12, color="#222222"),
        plot_bgcolor="#f5f5f5",
    )
    return fig

@register()
def draw_season_mcp_plot(
    relocations_gdf: AnyGeoDataFrame,
    seasons_df: AnyDataFrame,
    season_colors: Union[dict[str,str]|None]=None,
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

    x_min = relocations_gdf["fixtime"].min()
    x_max = relocations_gdf["fixtime"].max()
    relocations_gdf = Relocations.from_gdf(relocations_gdf)
    figure = mcp(relocations_gdf)
    figure = add_seasons_square(
        fig =figure, 
        season_df = seasons_df,
        show_labels=False,
        colors=season_colors,
        show_legend=False
    )
    figure.update_xaxes(range=[x_min, x_max])
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

def collar_event_timeline_plot(
    geodataframe: AnyGeoDataFrame,
    collar_events: Optional[AnyDataFrame] = None,
) -> go.FigureWidget:
    fig = go.FigureWidget()
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

@register()
def draw_season_collared_plot(
    events_gdf: AnyDataFrame,
    relocations_gdf: AnyGeoDataFrame,
    seasons_df: AnyDataFrame,
    season_colors: Union[dict[str,str]|None]=None,
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
    subject_name = relocations_gdf["subject_name"].unique()[0]
    if events_gdf is None or events_gdf.empty:
        print(f"No events data for subject '{subject_name}'.")
        events_gdf = None
    else:
        events_gdf = events_gdf[events_gdf["subject_name"] == subject_name]
        if events_gdf.empty:
            print(f"No events found for subject '{subject_name}'.")
            events_gdf = None
    x_min = relocations_gdf["fixtime"].min()
    x_max = relocations_gdf["fixtime"].max()
    figure = collar_event_timeline_plot(relocations_gdf, events_gdf)
    figure = add_seasons_square(
            fig =figure, 
            season_df = seasons_df,
            show_labels=False,
            colors=season_colors,
            show_legend=False
        )
    figure.update_xaxes(range=[x_min, x_max])
    return figure.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))

import re
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Literal
from pydantic import Field
from matplotlib import colormaps
from matplotlib.colors import to_hex
from typing import Annotated, Any, cast
from ecoscope_workflows_core.decorators import task
from ecoscope.base.utils import hex_to_rgba  # type: ignore[import-untyped]
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_custom.tasks.spatial_ops import spatial_join
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame, EmptyDataFrame
from ecoscope_workflows_ext_custom.tasks.transformation._color_utils import ColorPalette, CustomPalette
from ecoscope_workflows_ext_custom.tasks.io._spatial_features import EarthRangerSource
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerClient
from pydantic.json_schema import SkipJsonSchema

REQUIRED_COLUMNS = ["cell_id", "last_visited", "first_visited"]
_MC_SCHEME = {
    "equal_interval": "equalinterval",
    "quantile": "quantiles",
    "natural_breaks": "naturalbreaks",
    "fisher_jenks": "fisherjenks",
    "std_mean": "stdmean",
}


@task
def retrieve_spatial_features(
    client: EarthRangerClient,
    source: Annotated[EarthRangerSource | SkipJsonSchema[None], Field(default=None)] = None,
) -> AnyGeoDataFrame | EmptyDataFrame:
    """Fetch raw, unstyled spatial features from EarthRanger (EPSG:4326)."""
    print(f"[retrieve_spatial_features] source: {source}")
    if source is None:
        print("[retrieve_spatial_features] source is None, returning empty DataFrame")
        return cast(EmptyDataFrame, pd.DataFrame())
    source_obj = EarthRangerSource.model_validate(source)
    gdf = source_obj.get(client)
    if gdf.empty:
        print("[retrieve_spatial_features] result is empty, returning empty DataFrame")
        return cast(EmptyDataFrame, pd.DataFrame())
    print(f"[retrieve_spatial_features] result shape: {gdf.shape}, CRS: {gdf.crs}, columns: {list(gdf.columns)}")
    return cast(AnyGeoDataFrame, gdf)


@task
def add_time_since_visit(
    df: AnyDataFrame,
    time_range: TimeRange,
) -> AnyDataFrame:
    """Add elapsed-time columns measured from each row's last visit.

    Computes the gap between `time_range.since` and `last_visited`,
    then writes it out as `hours_since_visit` and `days_since_visit`.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input frame. Must contain the columns: cell_id, last_visited,
        first_visited.
    time_range : TimeRange
        The reference range; `time_range.since` is the anchor point
        elapsed time is measured from.

    Returns
    -------
    GeoDataFrame
        A copy with `hours_since_visit` and `days_since_visit` added.

    Raises
    ------
    ValueError
        If any required column is missing from `gdf`.
    """
    print(f"[add_time_since_visit] input shape: {df.shape}, columns: {list(df.columns)}")
    print(f"[add_time_since_visit] time_range.until: {time_range.until}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Frame is missing required column(s): {missing}. " f"Expected: {REQUIRED_COLUMNS}.")
    since = time_range.until - df["last_visited"]
    df["hours_since_visit"] = since.dt.total_seconds() / 3600
    df["days_since_visit"] = df["hours_since_visit"] / 24
    lo = df["days_since_visit"].min()
    hi = df["days_since_visit"].max()
    print(f"[add_time_since_visit] output shape: {df.shape}, days_since_visit range: [{lo:.2f}, {hi:.2f}]")
    return df


def _compute_edges(vals: pd.Series, scheme: str, bins: int) -> np.ndarray:
    """Bin edges from a mapclassify scheme, clipped to [vals.min(), vals.max()]."""
    import mapclassify

    name = _MC_SCHEME.get(scheme)
    if name is None:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose from {list(_MC_SCHEME)}.")

    lo, hi = float(vals.min()), float(vals.max())
    y = vals.to_numpy()
    # std_mean is defined by std multiples, not a class count, so k doesn't apply
    mc = mapclassify.classify(y, name) if name == "stdmean" else mapclassify.classify(y, name, k=bins)

    # keep only breaks strictly inside the data range (drops std_mean's out-of-range edges)
    interior = [b for b in np.asarray(mc.bins, dtype=float) if lo < b < hi]
    return np.unique(np.array([lo, *interior, hi], dtype=float))


@task
def add_visit_bins(
    df: AnyDataFrame,
    col: str,
    visited_col: str,
    new_col: str,
    bins: int = 5,
    scheme: Literal["equal_interval", "quantile", "natural_breaks", "fisher_jenks", "std_mean"] = "natural_breaks",
) -> AnyDataFrame:
    print(
        f"[add_visit_bins] input shape: {df.shape}, col: {col!r}, visited_col: {visited_col!r}, "
        f"new_col: {new_col!r}, bins: {bins}, scheme: {scheme!r}"
    )
    # visited AND has a real value
    mask = df[visited_col].fillna(False) & df[col].notna()
    vals = df.loc[mask, col].abs()
    print(f"[add_visit_bins] visited rows: {mask.sum()}, unvisited rows: {(~mask).sum()}")

    df[new_col] = "Unvisited"

    def fmt(x):
        return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")

    if vals.empty:
        labels = []
    elif vals.nunique() == 1:
        labels = [fmt(vals.iloc[0])]
        df.loc[mask, new_col] = labels[0]
    else:
        edges = np.unique(np.round(_compute_edges(vals, scheme, bins), 2))
        # don't let rounding pull the outer edges inward and orphan min/max
        edges[0] = min(edges[0], vals.min())
        edges[-1] = max(edges[-1], vals.max())

        if len(edges) < 2:  # everything collapsed to one edge
            labels = [fmt(vals.iloc[0])]
            df.loc[mask, new_col] = labels[0]
        else:
            labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(len(edges) - 1)]
            binned = pd.cut(vals, bins=edges, labels=labels, include_lowest=True)
            df.loc[mask, new_col] = binned.astype("object").where(binned.notna(), "Unvisited")

    df[new_col] = pd.Categorical(df[new_col], categories=["Unvisited"] + labels, ordered=True)
    print(f"[add_visit_bins] bin labels: {['Unvisited'] + labels}")
    return df


@task
def add_bin_colors(df: AnyDataFrame, col: str, new_col: str, cmap: ColorPalette, unvisited="#808080"):
    print(
        f"[add_bin_colors] input shape: {df.shape}, col: {col!r}, "
        f"new_col: {new_col!r}, cmap: {cmap!r}, unvisited: {unvisited!r}"
    )
    # use the categorical's own order if present, else sorted uniques
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        cats = list(df[col].cat.categories)
    else:
        cats = sorted(df[col].dropna().unique())

    ramp_cats = [c for c in cats if c != "Unvisited"]

    if isinstance(cmap, CustomPalette):
        colors = cmap.colors
        color_map = {c: colors[i % len(colors)] for i, c in enumerate(ramp_cats)}
    else:
        cm = colormaps[cmap.name]
        # evenly spaced samples across the colormap, one per non-Unvisited bin
        points = np.linspace(0, 1, len(ramp_cats)) if len(ramp_cats) > 1 else [0.5]
        color_map = {c: to_hex(cm(p)) for c, p in zip(ramp_cats, points)}
    color_map["Unvisited"] = unvisited

    df[new_col] = df[col].map(color_map)
    print(f"[add_bin_colors] color map: {color_map}")
    return df


@task
def add_non_null_flag(
    df: AnyDataFrame,
    source_column : str,
    flag_column: str = "is_present",
) -> AnyDataFrame:
    """Add a boolean column flagging where `source_column` is not null.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input frame.
    source_column : str
        Column to test for non-null values.
    flag_column : str, default "is_present"
        Name of the boolean column to create.

    Returns
    -------
    GeoDataFrame
        A copy with the boolean flag column added.

    Raises
    ------
    ValueError
        If `source_column` is missing from `gdf`.
    """
    print(
        f"[add_non_null_flag] input shape: {df.shape}, "
        f"source_column: {source_column!r}, flag_column: {flag_column!r}"
    )
    if source_column not in df.columns:
        raise ValueError(f"Frame is missing required column {source_column!r}.")

    df = df.copy()
    df[flag_column] = df[source_column].notna()
    non_null_count = df[flag_column].sum()
    print(f"[add_non_null_flag] non-null in {source_column!r}: {non_null_count} / {len(df)}")
    return df









@task
def compute_dwell_time(
    patrol_trajectories: AnyGeoDataFrame,
    gridded_spatial_feature: AnyGeoDataFrame,
) -> AnyDataFrame:
    """Compute time spent per grid cell from patrol trajectory segments.

    Moving segments are split at cell boundaries and their duration is
    apportioned by the fraction of segment length in each cell.
    Stationary segments (zero length) contribute their whole duration
    to the containing cell.

    Parameters
    ----------
    patrol_trajectories : GeoDataFrame
        Trajectory segments. Must contain `id`, `timespan_seconds`,
        and a geometry column.
    gridded_spatial_feature : GeoDataFrame
        Grid cells. Must contain `cell_id` and a geometry column.

    Returns
    -------
    DataFrame
        One row per `cell_id` with `seconds_in_cell`,
        `minutes_in_cell`, and `hours_in_cell`.
    """
    print(f"[compute_dwell_time] trajectories shape: {patrol_trajectories.shape}, CRS: {patrol_trajectories.crs}")
    print(f"[compute_dwell_time] grid shape: {gridded_spatial_feature.shape}, CRS: {gridded_spatial_feature.crs}")
    tracks = patrol_trajectories.copy()
    tracks["segment_length"] = tracks.geometry.length

    moving = tracks[tracks["segment_length"] > 0].copy()
    still = tracks[tracks["segment_length"] == 0].copy()
    print(f"[compute_dwell_time] moving segments: {len(moving)}, stationary segments: {len(still)}")

    # moving: split at cell borders, apportion time by length fraction
    pieces = overlay_gdf(
        moving[["id", "timespan_seconds", "segment_length", "geometry"]],
        gridded_spatial_feature[["cell_id", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    pieces["frac"] = pieces.geometry.length / pieces["segment_length"]
    pieces["time_in_cell"] = pieces["timespan_seconds"] * pieces["frac"]

    # stationary: whole duration goes to the containing cell
    if len(still):
        pts = still.copy()
        pts["geometry"] = pts.geometry.representative_point()
        pts = spatial_join(
            pts[["id", "timespan_seconds", "geometry"]],
            gridded_spatial_feature[["cell_id", "geometry"]],
            how="inner",
            predicate="within",
        )
        pts["time_in_cell"] = pts["timespan_seconds"]
        pieces = pd.concat(
            [pieces[["cell_id", "time_in_cell"]], pts[["cell_id", "time_in_cell"]]],
            ignore_index=True,
        )
    else:
        pieces = pieces[["cell_id", "time_in_cell"]]

    # aggregate per cell
    dwell = pieces.groupby("cell_id").agg(seconds_in_cell=("time_in_cell", "sum")).reset_index()
    dwell["minutes_in_cell"] = dwell["seconds_in_cell"] / 60
    dwell["hours_in_cell"] = dwell["seconds_in_cell"] / 3600
    lo = dwell["hours_in_cell"].min()
    hi = dwell["hours_in_cell"].max()
    print(f"[compute_dwell_time] output shape: {dwell.shape}, hours_in_cell range: [{lo:.2f}, {hi:.2f}]")
    return dwell


@task
def compute_patrol_effort_fraction(gdf: AnyGeoDataFrame) -> float:
    """Percentage of the gridded area that has been patrolled (visited).

    Returns a value in [0, 100] = (patrolled area / total area) * 100.
    Expects non-overlapping grid cells in a projected CRS.
    """
    print(f"[compute_patrol_effort_fraction] input shape: {gdf.shape}, CRS: {gdf.crs}")
    # 1. required column
    if "visit_bin" not in gdf.columns:
        raise KeyError("Expected a 'visit_bin' column in the GeoDataFrame.")

    # 2. CRS must be projected, or .area is in square degrees (meaningless)
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS; set one before computing area.")
    if gdf.crs.is_geographic:
        raise ValueError(
            f"CRS {gdf.crs.to_epsg()} is geographic. Reproject to a projected CRS "
            "(e.g. gdf.to_crs(gdf.estimate_utm_crs())) before calling this."
        )

    # 3. guard against an empty / zero-area frame
    total_area = gdf.geometry.area.sum()
    if total_area == 0:
        return 0.0

    patrolled_area = gdf.loc[gdf["visit_bin"] != "Unvisited", "geometry"].area.sum()
    fraction = patrolled_area / total_area
    percentage = round(fraction * 100, 2)
    print(
        f"[compute_patrol_effort_fraction] patrolled area: {patrolled_area:.2f}, "
        f"total area: {total_area:.2f}, coverage: {percentage:.2f}%"
    )
    return percentage


@task
def set_spatial_features_opacity(
    gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="Styled GeoDataFrame from get_spatial_features.", exclude=True),
    ],
    fill_opacity: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Fill opacity for polygon interiors. Set to 0 to show outlines only.",
        ),
    ] = 0.0,
    line_opacity: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Opacity of polygon borders from 0 (transparent) to 1 (fully opaque).",
        ),
    ] = 1.0,
) -> AnyGeoDataFrame:
    """Overwrite the alpha channel of fill and line color columns to control polygon styling."""
    gdf = gdf.copy()
    fill_alpha = int(fill_opacity * 255)
    line_alpha = int(line_opacity * 255)
    if "get_fill_color" in gdf.columns:
        gdf["get_fill_color"] = gdf["get_fill_color"].apply(
            lambda c: c[:3] + [fill_alpha] if isinstance(c, list) and len(c) >= 3 else c
        )
    if "get_line_color" in gdf.columns:
        gdf["get_line_color"] = gdf["get_line_color"].apply(
            lambda c: c[:3] + [line_alpha] if isinstance(c, list) and len(c) >= 3 else c
        )
    return cast(AnyGeoDataFrame, gdf)

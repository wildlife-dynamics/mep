import os
import io
import logging
import requests
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
from pyproj import Transformer
import concurrent.futures as cf
from pydantic import BaseModel,Field
from shapely.geometry import LineString
from typing import Annotated,Literal,cast
from pydantic.json_schema import SkipJsonSchema
from ecoscope_workflows_core.decorators import task
from ecoscope.base.utils import hex_to_rgba  # type: ignore[import-untyped]
from ecoscope_workflows_ext_ecoscope.schemas import TrajectoryGDF
from ecoscope_workflows_core.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.results._map import (
PydeckAnnotation,
PydeckString,
LayerStyleBase,
ColorAccessor,
FloatAccessor,
UnitType,
LegendDefinition,
LayerDefinition,
LegendStyle,
ViewState, 
LegendFromDataframe,
PYDECK_CUSTOM_LIBRARIES,
_model_dump_with_pydeck_literals,
LegendSegment,
BitmapLayerDefinition,
view_state_from_layers
)

logger = logging.getLogger(__name__)

TILE = 256
DEFAULT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRARIUM_ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}
DEFAULT_TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
SURFACE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

TerrainStrategy = Annotated[
    Literal["best-available", "no-overlap", "never"], PydeckAnnotation
]

class TimelineAnimation(BaseModel):
    """Settings for an animated TripsLayer timeline.

    Either construct directly, or derive from data with
    timeline_animation_from_gdf(). All time fields are in the same
    units as the layer's timestamps (typically seconds).
    """
    fade_ratio: float = Field(
        default=0.05,
        gt=0,
        le=1,
        description="Comet-tail length as a fraction of the total time span (0–1].",
    )
    animation_speed: float = Field(
        default=10000.0,
        gt=0,
        description="Amount currentTime advances per tick (per-frame increment).",
    )
    fps_limit: float = Field(
        default=20.0,
        gt=0,
        description="Maximum animation frames per second.",
    )
    
class TerrainLayerDefinition(BaseModel):
    """A 3D terrain layer built from RGB-encoded elevation tiles, optionally draped with a texture.
    See https://deck.gl/docs/api-reference/geo-layers/terrain-layer for more info."""

    elevation_data: Annotated[
        PydeckString,
        Field(description="URL template (or single image) for the RGB-encoded elevation tiles."),
    ] = DEFAULT_TERRAIN_URL
    texture: Annotated[
        PydeckString | SkipJsonSchema[None],
        AdvancedField(default=None, description="URL template for tiles draped over the terrain."),
    ] = SURFACE,
    elevation_decoder: Annotated[
        dict, AdvancedField(default=lambda: dict(TERRARIUM_ELEVATION_DECODER))
    ] = TERRARIUM_ELEVATION_DECODER  # type: ignore[assignment]
    wireframe: Annotated[bool, AdvancedField(default=False)] = False
    elevation_scale: Annotated[float, AdvancedField(default=1)] = 1
    min_zoom: Annotated[int, AdvancedField(default=0)] = 0
    max_zoom: Annotated[int, AdvancedField(default=15)] = 15
    strategy: Annotated[TerrainStrategy, AdvancedField(default="no-overlap")] = "no-overlap"
    opacity: Annotated[float, AdvancedField(default=1, ge=0, le=1)] = 1
    
class TripsLayerStyle(LayerStyleBase):
    """
    Trips Layer style kwargs
    See https://deck.gl/docs/api-reference/geo-layers/trips-layer for more info
    """

    get_path: Annotated[str, AdvancedField(default="geometry.coordinates")] = (
        "geometry.coordinates"
    )
    get_timestamps: Annotated[str, AdvancedField(default="timestamps")] = "timestamps"
    get_color: Annotated[
        ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None
    get_width: Annotated[
        FloatAccessor | SkipJsonSchema[None], AdvancedField(default=1)
    ] = 1
    width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    width_scale: Annotated[float, AdvancedField(default=1)] = 1
    width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    width_max_pixels: Annotated[
        float | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None
    cap_rounded: Annotated[bool, AdvancedField(default=True)] = True
    joint_rounded: Annotated[bool, AdvancedField(default=False)] = False
    billboard: Annotated[bool, AdvancedField(default=False)] = False
    fade_trail: Annotated[bool, AdvancedField(default=True)] = True
    
class TerrainSampling(BaseModel):
    """Per-vertex ground-elevation sampling for 3D trips draped over a TerrainLayer.

    Pass `terrain=None` to trajectory_to_trips for flat (z=0) paths and skip the network.
    """

    offset: float = Field(
        default=30.0, description="Metres added above the sampled ground at every vertex."
    )
    zoom: int = Field(
        default=12, description="Terrarium tile zoom used for elevation sampling."
    )
    vertical_scale: float = Field(
        default=1.0,
        description="Must equal the TerrainLayer's elevation_scale so trips align with the mesh.",
    )
    ground_elevation: float = Field(
        default=1000.0, description="Constant ground used only if DEM sampling fails."
    )
    cache_dir: str | None = Field(
        default=None, description="Optional dir to cache DEM tiles so reruns skip the network."
    )

@task
def create_terrain_layer(
    elevation_data: Annotated[
        str, Field(description="URL template for RGB-encoded elevation tiles.")
    ] = DEFAULT_TERRAIN_URL,
    texture: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL template for tiles draped over the terrain."),
    ] = None,
    elevation_scale: Annotated[
        float, Field(description="Vertical exaggeration factor.")
    ] = 1,
    extruded: Annotated[bool, AdvancedField(default=True)] = True,
    wireframe: Annotated[bool, AdvancedField(default=False)] = False,
    min_zoom: Annotated[int, AdvancedField(default=0)] = 0,
    max_zoom: Annotated[int, AdvancedField(default=15)] = 15,
    elevation_decoder: Annotated[
        dict | SkipJsonSchema[None],
        AdvancedField(default=None, description="RGB->elevation decoder. Defaults to Terrarium."),
    ] = None,
) -> Annotated[TerrainLayerDefinition, Field()]:
    """Creates a terrain layer definition from elevation tiles (+ optional texture)."""
    return TerrainLayerDefinition(
        elevation_data=elevation_data,
        texture=texture,
        elevation_scale=elevation_scale,
        extruded=extruded,
        wireframe=wireframe,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        elevation_decoder=elevation_decoder or dict(TERRARIUM_ELEVATION_DECODER),
    )
    
@task
def create_trips_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        TripsLayerStyle | SkipJsonSchema[None],
        AdvancedField(
            default=TripsLayerStyle(), description="Style arguments for the layer."
        ),
    ] = None,
    trail_frac: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If set, trail_length = span * trail_frac, computed from the data.",
        ),
    ] = None,
    current_frac: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If set, current_time = span * current_frac, computed from the data.",
        ),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates an animated trips layer definition based on the provided configuration.

    If trail_frac and/or current_frac are provided and a geodataframe is given,
    trail_length and current_time are derived from the timeline span of the
    timestamps column referenced by layer_style.get_timestamps.
    """
    layer_style = layer_style or TripsLayerStyle()

    if (trail_frac is not None or current_frac is not None) and geodataframe is not None:
        ts_col = layer_style.get_timestamps
        timestamps = (t for ts in geodataframe[ts_col] for t in ts)
        try:
            span = max(timestamps)
        except ValueError:
            span = None  # no timestamps -> leave style values untouched

        if span is not None:
            if trail_frac is not None:
                layer_style.trail_length = round(span * trail_frac, 2)
            if current_frac is not None:
                layer_style.current_time = round(span * current_frac, 2)

    return LayerDefinition(
        layer_type="TripsLayer",
        layer_style=layer_style,
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
    )
    
def _build_map_deck(
    geo_layers,
    tile_layers,
    static,
    title,
    legend_style,
    max_zoom,
    view_state,
    widget_id,
    extra_widgets=None,
):
    """Builds and returns the pdk.Deck (without rendering). Shared by draw_map
    and draw_animated_map. `extra_widgets` are appended after the defaults."""
    import pydeck as pdk  # type: ignore[import-untyped]

    pdk.settings.custom_libraries = PYDECK_CUSTOM_LIBRARIES

    DEFAULT_WIDGETS = [
        pdk.Widget("ScaleWidget", placement="bottom-left", id="ScaleWidget"),
        pdk.Widget("SaveImageWidget", placement="top-right", id="SaveImageWidget"),
    ]

    tile_layers = [] if tile_layers is None else list(tile_layers)
    if legend_style is None:
        legend_style = LegendStyle()

    legend_values: list = []
    map_layers: list = []
    map_widgets: list = DEFAULT_WIDGETS.copy()

    for tile_layer in tile_layers:
        if hasattr(tile_layer, "elevation_data"):          # TerrainLayerDefinition
            map_layers.append(
                pdk.Layer("TerrainLayer", **_model_dump_with_pydeck_literals(tile_layer))
            )
            if tile_layer.max_zoom < max_zoom:
                max_zoom = tile_layer.max_zoom
        elif isinstance(tile_layer, BitmapLayerDefinition):
            dump = _model_dump_with_pydeck_literals(tile_layer)
            dump.pop("legend", None)
            map_layers.append(pdk.Layer("BitmapLayer", **dump))
            if tile_layer.legend is not None:
                legend_values.append(tile_layer.legend)
        else:  # TiledBitmapLayerDefinition
            map_layers.append(
                pdk.Layer(
                    "TiledBitmapLayer",
                    data=tile_layer.url,
                    max_zoom=tile_layer.max_zoom,
                    min_zoom=tile_layer.min_zoom,
                    opacity=tile_layer.opacity,
                    tile_size=256,
                    widget_id=pdk.types.String(widget_id),
                )
            )
            if tile_layer.max_zoom < max_zoom:
                max_zoom = tile_layer.max_zoom

    if geo_layers is None:
        geo_layers = []
    elif isinstance(geo_layers, LayerDefinition):
        geo_layers = [geo_layers]

    for layer_def in geo_layers:
        if layer_def.data_url is not None:
            data = pdk.types.String(layer_def.data_url)
        elif layer_def.geodataframe is not None:
            gdf = layer_def.geodataframe.to_crs("EPSG:4326")
            is_multi = gdf.geometry.geom_type == "MultiPolygon"
            if is_multi.any():
                gdf = pd.concat(
                    [gdf[~is_multi], gdf[is_multi].explode(index_parts=False)],
                    ignore_index=True,
                )
            data = gdf

        map_layers.append(
            pdk.Layer(
                type=layer_def.layer_type,
                data=data,
                **_model_dump_with_pydeck_literals(layer_def.layer_style),
            )
        )

        if legend_def := layer_def.legend:
            if isinstance(legend_def, LegendSegment):
                legend_values.append(legend_def)
            elif isinstance(legend_def, LegendFromDataframe):
                if layer_def.geodataframe is not None:
                    legend_values.append(
                        legend_def.build_legend_from_dataframe(layer_def.geodataframe)
                    )
                else:
                    logger.warning(
                        "LegendFromDataframe legend skipped for layer '%s': "
                        "no geodataframe is available (layer uses data_url).",
                        layer_def.layer_type,
                    )

    if legend_values:
        map_widgets.append(
            pdk.Widget(
                "LegendWidget",
                id="LegendWidget",
                legend_values=legend_values,
                placement=legend_style.placement,
            )
        )
    if title:
        map_widgets.append(pdk.Widget("TitleWidget", id="TitleWidget", title=title))

    if extra_widgets:
        map_widgets.extend(extra_widgets)

    return pdk.Deck(
        layers=map_layers,
        widgets=map_widgets,
        initial_view_state=view_state
        or view_state_from_layers(layers=geo_layers, max_zoom=max_zoom),
        views=pdk.View("MapView", controller=not static, repeat=True),
        parameters={
            "depthTest": any(getattr(l, "extruded", False) for l in map_layers)
        },
        map_style=pdk.map_styles.LIGHT_NO_LABELS,
    )

def _make_session(pool=16):
    """A pooled, keep-alive session so repeated tile fetches reuse connections."""
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool, pool_maxsize=pool, max_retries=3
    )
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

def _decode_terrarium(content):
    img = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    return arr[:, :, 0] * 256 + arr[:, :, 1] + arr[:, :, 2] / 256 - 32768

def sample_elevations(
    lonlats,
    zoom=12,
    url=DEFAULT_TERRAIN_URL,
    session=None,
    _cache=None,
    max_workers=16,
    cache_dir=None,
):
    """Sample ground elevation (m) for an array of (lon, lat) points, bilinearly.

    Vectorised: all points are converted to pixel coords at once, every UNIQUE tile is
    fetched a single time (in parallel, over one pooled session), and the bilinear blend is
    done in NumPy grouped per tile. `_cache` (a dict) persists decoded tiles across calls;
    `cache_dir`, if given, also persists raw tiles on disk so reruns skip the network.
    """
    pts = np.asarray(lonlats, dtype=np.float64)
    if pts.size == 0:
        return []
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)

    sess = session or _make_session(pool=max_workers)
    cache = {} if _cache is None else _cache

    lon, lat = pts[:, 0], pts[:, 1]
    n = TILE * (2 ** zoom)
    px = (lon + 180.0) / 360.0 * n
    s = np.sin(np.radians(lat))
    py = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n

    tx = np.floor(px / TILE).astype(np.int64)
    ty = np.floor(py / TILE).astype(np.int64)
    fx = px - tx * TILE
    fy = py - ty * TILE
    unique_tiles = {(int(a), int(b)) for a, b in zip(tx.tolist(), ty.tolist())}
    to_fetch = [k for k in unique_tiles if k not in cache]

    def load(key):
        txx, tyy = key
        if cache_dir is not None:
            fp = os.path.join(cache_dir, f"{zoom}_{txx}_{tyy}.png")
            if os.path.exists(fp):
                with open(fp, "rb") as fh:
                    return key, _decode_terrarium(fh.read())
        r = sess.get(url.format(z=zoom, x=txx, y=tyy), timeout=30)
        r.raise_for_status()
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, f"{zoom}_{txx}_{tyy}.png"), "wb") as fh:
                fh.write(r.content)
        return key, _decode_terrarium(r.content)

    if to_fetch:
        with cf.ThreadPoolExecutor(max_workers=min(max_workers, len(to_fetch))) as ex:
            for key, dem in ex.map(load, to_fetch):
                cache[key] = dem

    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    x1 = np.minimum(x0 + 1, TILE - 1)
    y1 = np.minimum(y0 + 1, TILE - 1)
    x0 = np.clip(x0, 0, TILE - 1)
    y0 = np.clip(y0, 0, TILE - 1)
    dx = fx - x0
    dy = fy - y0

    out = np.empty(len(pts), dtype=np.float64)
    for key in unique_tiles:
        dem = cache[key]
        m = (tx == key[0]) & (ty == key[1])
        ix0, iy0, ix1, iy1 = x0[m], y0[m], x1[m], y1[m]
        ddx, ddy = dx[m], dy[m]
        out[m] = (
            dem[iy0, ix0] * (1 - ddx) * (1 - ddy)
            + dem[iy0, ix1] * ddx * (1 - ddy)
            + dem[iy1, ix0] * (1 - ddx) * ddy
            + dem[iy1, ix1] * ddx * ddy
        )
    return out.tolist()

@task
def trajectory_to_trips(
    trajectory_gdf: TrajectoryGDF,
    subject_name_col: Annotated[
        str, Field(description="Column holding the subject name.")
    ] = "subject__name",
    subject_hex_col: Annotated[
        str, Field(description="Column holding the subject hex color.")
    ] = "subject__hex",
    terrain: Annotated[
        TerrainSampling | None,
        Field(description="Elevation sampling config. None -> flat ground (z=0)."),
    ] = None,
) -> AnyGeoDataFrame:
    # The DEM is indexed in lon/lat, and deck wants lon/lat too, so work in WGS84.
    src_crs = trajectory_gdf.crs
    to_wgs84 = None
    if src_crs is not None and src_crs.to_epsg() != 4326:
        to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    # --- Pass 1: build each subject's stitched (lon, lat) track + timestamps --------
    # Defer z entirely so we can sample ALL points in one vectorised, batched call.
    tracks = []
    all_lonlats = []  # master point buffer; tracks index into it via [i0:i1]
    for group_key, g in trajectory_gdf.groupby("groupby_col"):
        g = g.sort_values("segment_start")
        coords2d, times = [], []
        for _, r in g.iterrows():
            xy = np.asarray(r.geometry.coords, dtype=np.float64)
            if to_wgs84 is not None:
                lon, lat = to_wgs84.transform(xy[:, 0], xy[:, 1])  # vectorised transform
                seg = list(zip(np.asarray(lon).tolist(), np.asarray(lat).tolist()))
            else:
                seg = list(zip(xy[:, 0].tolist(), xy[:, 1].tolist()))
            n = len(seg)
            s, e = r.segment_start.timestamp(), r.segment_end.timestamp()
            ts = [s] if n == 1 else [s + (e - s) * i / (n - 1) for i in range(n)]
            start = 1 if coords2d and coords2d[-1] == seg[0] else 0
            coords2d += seg[start:]
            times += ts[start:]
        if len(coords2d) < 2:  # LineString needs >= 2 vertices
            continue
        i0 = len(all_lonlats)
        all_lonlats.extend(coords2d)
        tracks.append({
            "groupby_col": group_key,
            "i0": i0,
            "i1": len(all_lonlats),
            "raw_ts": times,
            "color": hex_to_rgba(g[subject_hex_col].iloc[0]),
            "name": g[subject_name_col].iloc[0],
        })

    # --- One batched elevation sample for the entire job ----------------------------
    if terrain is not None and all_lonlats:
        try:
            elevs = sample_elevations(
                all_lonlats, zoom=terrain.zoom, cache_dir=terrain.cache_dir
            )
            zs_all = [e * terrain.vertical_scale + terrain.offset for e in elevs]
        except Exception as exc:  # network/tile failure -> safe constant fallback
            logger.warning(
                "trajectory_to_trips: terrain sampling failed (%s); "
                "using constant ground_elevation=%s",
                exc, terrain.ground_elevation,
            )
            zs_all = [terrain.ground_elevation + terrain.offset] * len(all_lonlats)
    else:
        zs_all = [0.0] * len(all_lonlats)  # flat ground

    # --- Pass 2: attach z back to each subject's coordinates ------------------------
    for t in tracks:
        pts = all_lonlats[t["i0"]:t["i1"]]
        zz = zs_all[t["i0"]:t["i1"]]
        t["coordinates"] = [[lon, lat, z] for (lon, lat), z in zip(pts, zz)]
        del t["i0"], t["i1"]

    trips = pd.DataFrame(tracks)
    if trips.empty:
        return gpd.GeoDataFrame(
            columns=["groupby_col", "color", "name", "timestamps", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    # draw_animated_map owns the shared-clock normalization, so emit faithful timestamps
    trips["timestamps"] = trips["raw_ts"].apply(lambda ts: [t - ts[0] for t in ts])
    trips["geometry"] = trips["coordinates"].apply(LineString)
    trips = gpd.GeoDataFrame(trips, geometry="geometry", crs="EPSG:4326")
    # raw_ts/coordinates are now redundant (duplicated in timestamps/geometry)
    trips = trips.drop(columns=["raw_ts", "coordinates"])
    return cast(AnyGeoDataFrame, trips)

@task
def normalize_timestamps(df:AnyGeoDataFrame, target_span: int | None = None)->AnyGeoDataFrame:
    """Rescale all subjects onto one [0, COMMON_SPAN] timeline."""
    if df is None or len(df) == 0:
        return df

    # Preserve the originals so repeated calls stay idempotent.
    if "raw_ts" not in df.columns:
        df = df.copy()
        df["raw_ts"] = df["timestamps"].apply(lambda x: list(x))

    # Gather every timestamp across all rows.
    all_ts = []
    for ts in df["raw_ts"]:
        if isinstance(ts, (list, np.ndarray)):
            all_ts.extend(ts)
    if not all_ts:
        return df

    global_min, global_max = min(all_ts), max(all_ts)
    time_range = global_max - global_min
    if time_range <= 0:
        return df

    if target_span is None:
        common_span = max(1_000_000, min(15_000_000, int(time_range * 8)))
        if common_span >= 10_000_000:
            common_span = round(common_span / 1_000_000) * 1_000_000
        elif common_span >= 1_000_000:
            common_span = round(common_span / 500_000) * 500_000
        else:
            common_span = round(common_span / 100_000) * 100_000
    else:
        common_span = target_span

    print(f"time range {time_range/3600:.1f} h -> COMMON_SPAN {common_span:,}")

    df["timestamps"] = df["raw_ts"].apply(
        lambda raw: ((np.array(raw) - global_min) / time_range * common_span).tolist()
    )
    return df

@task
def draw_animated_map(
    geo_layers: Annotated[
        LayerDefinition | list[LayerDefinition] | SkipJsonSchema[None],
        Field(description="Map layers; must include one TripsLayer.", exclude=True),
    ] = None,
    tile_layers: Annotated[
        list | SkipJsonSchema[None],
        Field(description="Base maps and/or overlays, as in draw_map."),
    ] = None,
    animation: Annotated[
        TimelineAnimation | SkipJsonSchema[None],
        AdvancedField(default=TimelineAnimation(), description="Timeline settings."),
    ] = None,
    static: Annotated[bool, Field(default=False)] = False,
    title: Annotated[str | SkipJsonSchema[None], AdvancedField(default="")] = None,
    legend_style: Annotated[
        LegendStyle | SkipJsonSchema[None], AdvancedField(default=LegendStyle())
    ] = None,
    max_zoom: Annotated[int, AdvancedField(default=20)] = 20,
    view_state: Annotated[
        ViewState | SkipJsonSchema[None], AdvancedField(default=ViewState())
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None], Field(default=None, exclude=True)
    ] = None,
) -> Annotated[str, Field()]:
    """Like draw_map, but animates the TripsLayer with an interactive TimelineWidget.

    Returns a static HTML string (same contract as draw_map). The timeline range is
    derived from the trips data; the widget's scrubber and play button drive the
    layer's currentTime via an injected onTimeChange bridge.
    """
    import pydeck as pdk  # type: ignore[import-untyped]
    import numpy as np

    if animation is None:
        animation = TimelineAnimation()

    geo_list = (
        [geo_layers]
        if isinstance(geo_layers, LayerDefinition)
        else list(geo_layers or [])
    )
    trips_def = next(
        (ld for ld in geo_list if ld.layer_type == "TripsLayer"), None
    )
    if trips_def is None:
        raise ValueError(
            "draw_animated_map requires a TripsLayer in geo_layers "
            "(create one with create_trips_layer)."
        )

    style = getattr(trips_def, "layer_style", None)
    current_time = getattr(style, "current_time", 0.0) if style else 0.0

    fade_ratio      = animation.fade_ratio
    animation_speed = animation.animation_speed
    fps_limit       = animation.fps_limit

    gdf = trips_def.geodataframe
    all_ts = []
    if "timestamps" in gdf.columns:
        for ts in gdf["timestamps"]:
            if isinstance(ts, (list, np.ndarray)):
                all_ts.extend(ts)

    if all_ts:
        max_ts = max(all_ts)
        span = max_ts * 1.02
        trail_length = max_ts * 1.15
    else:
        # Fallbacks so the JS replacements always have valid numbers.
        max_ts = 0
        span = 1.0
        trail_length = fade_ratio

    print(f"max_ts : {max_ts} span: {span} trail_length: {trail_length} ")
    
    # Guard: a starting time past the span means nothing would animate.
    if current_time >= span:
        current_time = 0.0
    
    deck = _build_map_deck(
        geo_list,
        tile_layers,
        static,
        title,
        legend_style,
        max_zoom,
        view_state,
        widget_id,
    )

    html_str = deck.to_html(as_string=True)
    html_str = html_str.replace("const jsonInput =", "window.jsonInput =")
    html_str = html_str.replace(
        "const deckInstance = createDeck(",
        "window.deckInstance = createDeck(",
    )

    animation_js = """
<script>
// === Full Trail Visible - No disappearing start at the end ===
let currentTime   = __CURRENT_TIME__;
const maxTime     = __SPAN__;
const trailLength = __TRAIL_LENGTH__;
let animationSpeed = __ANIMATION_SPEED__;
let isPlaying     = true;
let lastFrameTime = 0;
const fpsInterval = 1000 / __FPS_LIMIT__;

let baseLayers = null;
let tripsBase  = null;
let tripsId    = null;

function buildLayers() {
  let effectiveTrailLength = trailLength;
    
  let effectiveOpacity = 0.95;

  // When we reach the end, use full trail length so nothing disappears
  if (currentTime >= maxTime * 0.995) {
    effectiveTrailLength = maxTime * 1.20;   // slightly longer than total
    effectiveOpacity = 0.98;
  }

  const trail = tripsBase.clone({
    id: tripsId + '-drawing',
    currentTime: currentTime,
    trailLength: effectiveTrailLength,
    fadeTrail: true,
    opacity: effectiveOpacity,
  });

  const out = [];
  baseLayers.forEach(function (l) {
    if (l.id === tripsId) {
      out.push(trail);
    } else {
      out.push(l);
    }
  });
  return out;
}

function frame(timestamp) {
  if (!isPlaying || !window.deckInstance) return;

  if (!timestamp) timestamp = 0;
  if (timestamp - lastFrameTime < fpsInterval) {
    requestAnimationFrame(frame);
    return;
  }
  lastFrameTime = timestamp;

  if (currentTime < maxTime) {
    currentTime = Math.min(maxTime, currentTime + animationSpeed);
  } else {
    isPlaying = false;        // Stop animation at the end
  }

  window.deckInstance.setProps({ layers: buildLayers() });
  requestAnimationFrame(frame);
}

// Start
const __startWhenReady = setInterval(function () {
  if (window.deckInstance && window.deckInstance.props && window.deckInstance.props.layers) {
    clearInterval(__startWhenReady);

    baseLayers = window.deckInstance.props.layers;
    tripsBase  = baseLayers.find(l => 'currentTime' in l.props);
    tripsId    = tripsBase.id;

    requestAnimationFrame(frame);
  }
}, 200);
</script>
"""
    animation_js = (
        animation_js
        .replace("__CURRENT_TIME__", str(current_time))
        .replace("__SPAN__", str(span))
        .replace("__TRAIL_LENGTH__", str(trail_length))
        .replace("__ANIMATION_SPEED__", str(animation_speed))
        .replace("__FPS_LIMIT__", str(fps_limit))
    )
    html_str = html_str.replace("</body>", animation_js + "</body>")
    return html_str
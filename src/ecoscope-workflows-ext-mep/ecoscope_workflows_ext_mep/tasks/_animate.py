import os
import io
import math
import base64
import pathlib
import logging
import requests
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
from pyproj import Transformer
import concurrent.futures as cf
from pydantic import BaseModel, Field
from shapely.geometry import LineString
from typing import Annotated, Literal, cast
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
    view_state_from_layers,
)

logger = logging.getLogger(__name__)

TILE = 256
DEFAULT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRARIUM_ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}
DEFAULT_TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
SURFACE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

TerrainStrategy = Annotated[Literal["best-available", "no-overlap", "never"], PydeckAnnotation]


class ScenegraphLayerDefinition(BaseModel):
    """An animated 3D head built from a glTF/GLB model (deck.gl ScenegraphLayer).

    Create one with create_scenegraph_layer() and pass it to
    draw_animated_map(head_layer=...). Its position and heading are driven per-frame
    from the TripsLayer, so it follows each subject's current location. `glb` accepts
    an http(s) URL, a `data:` URI, or a local file path (read and embedded as a data
    URI). None -> bundled default (the elephant model). If the ScenegraphLayer
    constructor or the glTF loader can't be resolved at runtime, the head silently
    falls back to the flat ScatterplotLayer dot.
    See https://deck.gl/docs/api-reference/mesh-layers/scenegraph-layer for more info.
    """

    enabled: Annotated[
        bool,
        AdvancedField(default=False, description="Enable the 3D head model. When off, subjects render as flat dots."),
    ] = False
    glb: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default="https://raw.githubusercontent.com/wildlife-dynamics/animate_subject_tracks/main/african_bush_elephant.glb",
            description="GLB source: an http(s) URL, a data: URI, or a local file path. "
            "None -> bundled default model (elephant).",
        ),
    ] = "https://raw.githubusercontent.com/wildlife-dynamics/animate_subject_tracks/main/african_bush_elephant.glb"
    size_scale: Annotated[
        float,
        AdvancedField(default=50.0, description="ScenegraphLayer sizeScale. Tune to your scene."),
    ] = 50.0
    size_min_pixels: Annotated[
        float,
        AdvancedField(
            default=12.0,
            description="Clamp the on-screen model to at least this many pixels so it stays visible when zoomed out.",
        ),
    ] = 12.0
    size_max_pixels: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(default=None, description="Optional upper clamp on the model's on-screen size in pixels."),
    ] = 75.0
    face_heading: Annotated[
        bool,
        AdvancedField(default=True, description="Rotate the model to face its direction of travel."),
    ] = True
    yaw_offset: Annotated[
        float,
        AdvancedField(
            default=0.0,
            description="Degrees added to the computed heading so the model's nose aligns with travel. "
            "Model-dependent; tweak if your model faces sideways.",
        ),
    ] = 0.0
    model_pitch: Annotated[
        float,
        AdvancedField(
            default=90.0,
            description="Tilt of the MODEL itself (deg), independent of the camera. "
            "Use to correct a model authored nose-up/down; NOT the view pitch.",
        ),
    ] = 90.0
    model_roll: Annotated[
        float,
        AdvancedField(
            default=0.0,
            description="Bank of the MODEL itself (deg), independent of the camera.",
        ),
    ] = 0.0
    smooth_samples: Annotated[
        int,
        AdvancedField(
            default=2,
            description="Heading/slope smoothing window in track fixes (+/- N). "
            "Higher = smoother orientation but more lag; 0 = raw single segment.",
        ),
    ] = 2
    terrain_pitch: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="Tilt the model to the terrain slope (from each fix's z) so it "
            "noses up on climbs / down on descents. OFF by default -> stays upright; "
            "enable only for steady, climbing tracks (it can tip near-stationary subjects).",
        ),
    ] = False
    terrain_pitch_scale: Annotated[
        float,
        AdvancedField(
            default=1.0,
            description="Sign/strength of terrain pitch. Set -1.0 to flip if the model "
            "tilts the wrong way; <1 to soften. (Tilt is also capped at +/-20deg.)",
        ),
    ] = 1.0
    min_move_m: Annotated[
        float,
        AdvancedField(
            default=3.0,
            description="If the subject moves less than this (m) across the smoothing "
            "window, hold the last heading and keep the model level -- stops it spinning "
            "or tipping while milling in place.",
        ),
    ] = 3.0
    pbr_lighting: Annotated[
        bool,
        AdvancedField(default=True, description="Physically-based lighting ('pbr'); False -> flat shading."),
    ] = True
    tint: Annotated[
        list[int] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Optional RGB tint over the model as [R, G, B]. None -> the model's own materials.",
        ),
    ] = [220, 220, 255]
    use_track_color: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Colour the model with each subject's track colour. "
            "False -> use `tint` (or the model's own materials). Note: the colour "
            "multiplies the model's material, so it reads truest with a light/neutral "
            "glb and flat lighting (pbr_lighting=False).",
        ),
    ] = True


def _resolve_glb_data_uri(glb: str | None) -> str:
    """Resolve a ScenegraphLayerDefinition.glb source to something deck.gl can load.

    URLs and data: URIs pass through; a local path is read and base64-embedded so the
    output HTML stays self-contained; None -> the bundled default (elephant) data URI.
    """
    if glb.startswith(("http://", "https://", "data:")):
        return glb
    raw = pathlib.Path(glb).read_bytes()
    return "data:model/gltf-binary;base64," + base64.b64encode(raw).decode()


class TimelineAnimation(BaseModel):
    """Settings for an animated TripsLayer timeline.

    Either construct directly, or derive from data with
    timeline_animation_from_gdf(). All time fields are in the same
    units as the layer's timestamps (typically seconds).
    """

    fade_ratio: float = Field(
        default=0.55,
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
        default=30.0,
        gt=0,
        description="Maximum animation frames per second.",
    )

    # --- Historic-track ("fade to white") trail ---------------------------------
    show_history: bool = Field(
        default=True,
        description="Draw the already-traversed path behind the comet (the 'fade to white' track).",
    )
    history_color: tuple[int, int, int] = Field(
        default=(255, 255, 255),
        description="RGB colour the historic track settles to. Default white.",
    )
    history_opacity: float = Field(default=0.85, ge=0, le=1, description="Opacity of the historic track.")
    fade_history: bool = Field(
        default=False,
        description="If True the historic track also fades by opacity along its length; "
        "if False it stays a solid line all the way back to the start.",
    )

    # --- Current-position head marker -------------------------------------------
    show_head: bool = Field(
        default=True,
        description="Draw a marker at each subject's current position (no historic track on this layer).",
    )
    head_radius: float = Field(default=6.0, gt=0, description="Head-marker radius in pixels.")
    head_color: tuple[int, int, int] | None = Field(
        default=None,
        description="RGB fill for the head marker. None -> use each subject's own colour.",
    )
    head_outline_color: tuple[int, int, int] = Field(
        default=(255, 255, 255), description="RGB outline colour for the head marker."
    )
    head_outline_width: float = Field(default=1.5, ge=0, description="Head-marker outline width in pixels.")
    auto_rotate_speed: Annotated[
        float,
        AdvancedField(
            default=0.0,
            description="Camera rotation speed in degrees per second while the animation plays. "
            "0 = off; positive = clockwise; negative = counter-clockwise.",
        ),
    ] = 0.0


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
    ] = (SURFACE,)
    elevation_decoder: Annotated[dict, AdvancedField(default=lambda: dict(TERRARIUM_ELEVATION_DECODER))] = (
        TERRARIUM_ELEVATION_DECODER  # type: ignore[assignment]
    )
    wireframe: Annotated[bool, AdvancedField(default=False)] = False
    min_zoom: Annotated[int, AdvancedField(default=0)] = 0
    max_zoom: Annotated[int, AdvancedField(default=15)] = 15
    strategy: Annotated[TerrainStrategy, AdvancedField(default="no-overlap")] = "no-overlap"
    mesh_max_error: Annotated[float, AdvancedField(default=4)] = 4
    material: Annotated[bool, AdvancedField(default=True)] = True


class TripsLayerStyle(LayerStyleBase):
    """
    Trips Layer style kwargs
    See https://deck.gl/docs/api-reference/geo-layers/trips-layer for more info
    """

    get_path: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    get_timestamps: Annotated[str, AdvancedField(default="timestamps")] = "timestamps"
    get_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_width: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=1)] = 1
    width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    width_scale: Annotated[float, AdvancedField(default=1)] = 1
    width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    width_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    cap_rounded: Annotated[bool, AdvancedField(default=False)] = False
    joint_rounded: Annotated[bool, AdvancedField(default=False)] = False
    billboard: Annotated[bool, AdvancedField(default=False)] = False
    fade_trail: Annotated[bool, AdvancedField(default=True)] = True
    current_time: Annotated[float, AdvancedField(default=0)] = 0
    trail_length: Annotated[float, AdvancedField(default=0)] = 120


class TerrainSampling(BaseModel):
    """Per-vertex ground-elevation sampling for 3D trips draped over a TerrainLayer.

    Pass `terrain=None` to trajectory_to_trips for flat (z=0) paths and skip the network.
    """

    offset: float = Field(default=30.0, description="Metres added above the sampled ground at every vertex.")
    zoom: int = Field(default=15, description="Terrarium tile zoom used for elevation sampling.")
    elevation_data: str = Field(
        default=DEFAULT_TERRAIN_URL,
        description="Elevation tile URL template. Must match the TerrainLayer's elevation_data.",
    )
    elevation_decoder: dict | None = Field(
        default=TERRARIUM_ELEVATION_DECODER,
        description=(
            "RGB->elevation decoder. Must match the TerrainLayer's elevation_decoder so "
            "sampled z aligns with the rendered mesh. None -> Terrarium default."
        ),
    )
    ground_elevation: float = Field(default=1000.0, description="Constant ground used only if DEM sampling fails.")
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
    ] = SURFACE,
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
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        wireframe=wireframe,
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
        AdvancedField(default=TripsLayerStyle(), description="Style arguments for the layer."),
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
    return LayerDefinition(
        layer_type="TripsLayer",
        layer_style=layer_style,
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
    )


@task
def create_scenegraph_layer(
    enabled: Annotated[
        bool,
        AdvancedField(default=False, description="Enable the 3D head model. When off, subjects render as flat dots."),
    ] = False,
    glb: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="GLB source: an http(s) URL, a data: URI, or a local file path. "
            "None -> bundled default model (elephant).",
        ),
    ] = None,
    size_scale: Annotated[
        float, AdvancedField(default=50.0, description="ScenegraphLayer sizeScale. Tune to your scene.")
    ] = 50.0,
    size_min_pixels: Annotated[float, AdvancedField(default=12.0)] = 12.0,
    size_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None,
    face_heading: Annotated[bool, AdvancedField(default=True)] = True,
    yaw_offset: Annotated[float, AdvancedField(default=0.0)] = 0.0,
    model_pitch: Annotated[float, AdvancedField(default=0.0)] = 0.0,
    model_roll: Annotated[float, AdvancedField(default=0.0)] = 0.0,
    smooth_samples: Annotated[int, AdvancedField(default=2)] = 2,
    terrain_pitch: Annotated[bool, AdvancedField(default=False)] = False,
    terrain_pitch_scale: Annotated[float, AdvancedField(default=1.0)] = 1.0,
    min_move_m: Annotated[float, AdvancedField(default=3.0)] = 3.0,
    pbr_lighting: Annotated[bool, AdvancedField(default=True)] = True,
    tint: Annotated[list[int] | SkipJsonSchema[None], AdvancedField(default=None)] = None,
    use_track_color: Annotated[bool, AdvancedField(default=True)] = True,
) -> Annotated[ScenegraphLayerDefinition, Field()]:
    """Create an animated 3D head layer from a glTF/GLB model.

    Pass the result to draw_animated_map(head_layer=...). The model is placed at each
    subject's current position and (optionally) rotated to face its direction of travel,
    driven per-frame from the TripsLayer. With glb=None it uses the bundled default
    (elephant). If ScenegraphLayer / the glTF loader can't be resolved in the browser,
    the head falls back to the flat ScatterplotLayer dot.
    """
    return ScenegraphLayerDefinition(
        enabled=enabled,
        glb=glb,
        size_scale=size_scale,
        size_min_pixels=size_min_pixels,
        size_max_pixels=size_max_pixels,
        face_heading=face_heading,
        yaw_offset=yaw_offset,
        model_pitch=model_pitch,
        model_roll=model_roll,
        smooth_samples=smooth_samples,
        terrain_pitch=terrain_pitch,
        terrain_pitch_scale=terrain_pitch_scale,
        min_move_m=min_move_m,
        pbr_lighting=pbr_lighting,
        tint=tint,
        use_track_color=use_track_color,
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
        if hasattr(tile_layer, "elevation_data"):  # TerrainLayerDefinition
            map_layers.append(pdk.Layer("TerrainLayer", **_model_dump_with_pydeck_literals(tile_layer)))
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
                    legend_values.append(legend_def.build_legend_from_dataframe(layer_def.geodataframe))
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
        initial_view_state=view_state or view_state_from_layers(layers=geo_layers, max_zoom=max_zoom),
        views=pdk.View("MapView", controller=not static, repeat=True),
        parameters={"depthTest": any(getattr(l, "extruded", False) for l in map_layers)},
        map_style=pdk.map_styles.LIGHT_NO_LABELS,
    )


def _make_session(pool=16):
    """A pooled, keep-alive session so repeated tile fetches reuse connections."""
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=3)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _decode_elevation(content, decoder=None):
    d = decoder or TERRARIUM_ELEVATION_DECODER
    img = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    return arr[:, :, 0] * d["rScaler"] + arr[:, :, 1] * d["gScaler"] + arr[:, :, 2] * d["bScaler"] + d["offset"]


def sample_elevations(
    lonlats,
    zoom=12,
    url=DEFAULT_TERRAIN_URL,
    session=None,
    _cache=None,
    max_workers=16,
    cache_dir=None,
    decoder=None,
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
    n = TILE * (2**zoom)
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
                    return key, _decode_elevation(fh.read(), decoder)
        r = sess.get(url.format(z=zoom, x=txx, y=tyy), timeout=30)
        r.raise_for_status()
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, f"{zoom}_{txx}_{tyy}.png"), "wb") as fh:
                fh.write(r.content)
        return key, _decode_elevation(r.content, decoder)

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
    subject_name_col: Annotated[str, Field(description="Column holding the subject name.")] = "subject_name",
    subject_hex_col: Annotated[str, Field(description="Column holding the subject hex color.")] = "subject_hex",
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
        tracks.append(
            {
                "groupby_col": group_key,
                "i0": i0,
                "i1": len(all_lonlats),
                "raw_ts": times,
                "color": hex_to_rgba(g[subject_hex_col].iloc[0]),
                "name": g[subject_name_col].iloc[0],
            }
        )

    # --- One batched elevation sample for the entire job ----------------------------
    if terrain is not None and all_lonlats:
        try:
            elevs = sample_elevations(
                all_lonlats,
                zoom=terrain.zoom,
                url=terrain.elevation_data,
                decoder=terrain.elevation_decoder,
                cache_dir=terrain.cache_dir,
            )
            zs_all = [e + terrain.offset for e in elevs]
        except Exception as exc:  # network/tile failure -> safe constant fallback
            logger.warning(
                "trajectory_to_trips: terrain sampling failed (%s); " "using constant ground_elevation=%s",
                exc,
                terrain.ground_elevation,
            )
            zs_all = [terrain.ground_elevation + terrain.offset] * len(all_lonlats)
    else:
        zs_all = [0.0] * len(all_lonlats)  # flat ground

    # --- Pass 2: attach z back to each subject's coordinates ------------------------
    for t in tracks:
        pts = all_lonlats[t["i0"] : t["i1"]]
        zz = zs_all[t["i0"] : t["i1"]]
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
def normalize_timestamps(df: AnyGeoDataFrame, target_span: int | None = None) -> AnyGeoDataFrame:
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
    legend_style: Annotated[LegendStyle | SkipJsonSchema[None], AdvancedField(default=LegendStyle())] = None,
    max_zoom: Annotated[int, AdvancedField(default=20)] = 20,
    view_state: Annotated[ViewState | SkipJsonSchema[None], AdvancedField(default=ViewState())] = None,
    widget_id: Annotated[str | SkipJsonSchema[None], Field(default=None, exclude=True)] = None,
    head_layer: Annotated[
        ScenegraphLayerDefinition,
        AdvancedField(
            default=ScenegraphLayerDefinition(),
            description="3D glTF/GLB head model settings. Enable with the 'enabled' checkbox.",
        ),
    ] = ScenegraphLayerDefinition(),
) -> Annotated[str, Field()]:
    """Like draw_map, but animates the TripsLayer with an interactive TimelineWidget.

    Returns a static HTML string (same contract as draw_map). The timeline range is
    derived from the trips data; the widget's scrubber and play button drive the
    layer's currentTime via an injected onTimeChange bridge.
    """
    import numpy as np

    if animation is None:
        animation = TimelineAnimation()

    geo_list = [geo_layers] if isinstance(geo_layers, LayerDefinition) else list(geo_layers or [])
    trips_def = next((ld for ld in geo_list if ld.layer_type == "TripsLayer"), None)
    if trips_def is None:
        raise ValueError(
            "draw_animated_map requires a TripsLayer in geo_layers " "(create one with create_trips_layer)."
        )

    style = getattr(trips_def, "layer_style", None)
    current_time = getattr(style, "current_time", 0.0) if style else 0.0

    fade_ratio = animation.fade_ratio
    animation_speed = animation.animation_speed
    fps_limit = animation.fps_limit

    gdf = trips_def.geodataframe
    all_ts = []
    if "timestamps" in gdf.columns:
        for ts in gdf["timestamps"]:
            if isinstance(ts, (list, np.ndarray)):
                all_ts.extend(ts)

    if all_ts:
        max_ts = max(all_ts)
        span = max_ts * 1.02
        # The colored comet is a SHORT tail driven by fade_ratio; the full traversed
        # path is carried by the (white) history trail beneath it.
        comet_trail = max(span * fade_ratio, 1.0)
        history_trail = span * 1.20  # >= span -> solid back to the start
    else:
        # Fallbacks so the JS replacements always have valid numbers.
        max_ts = 0
        span = 1.0
        comet_trail = fade_ratio
        history_trail = 1.0

    print(f"max_ts : {max_ts} span: {span} comet_trail: {comet_trail} " f"history_trail: {history_trail}")

    # Guard: a starting time past the span means nothing would animate.
    if current_time >= span:
        current_time = 0.0

    if view_state is None:
        view_state = view_state_from_layers(layers=geo_list, max_zoom=max_zoom)

    # --- Marker / history params handed to the injected JS ------------------------
    def _rgb(c):
        return "[" + ", ".join(str(int(x)) for x in c) + "]"

    show_history = bool(animation.show_history)
    history_color_js = _rgb(animation.history_color)
    history_opacity = float(animation.history_opacity)
    fade_history_js = "true" if animation.fade_history else "false"

    show_head = bool(animation.show_head)
    head_radius = float(animation.head_radius)
    head_color_js = "null" if animation.head_color is None else _rgb(animation.head_color)
    head_outline_js = _rgb(animation.head_outline_color)
    head_outline_width = float(animation.head_outline_width)
    auto_rotate_speed = float(animation.auto_rotate_speed)

    # --- Optional 3D head model (ScenegraphLayer via head_layer) ----------------
    hm = head_layer
    if hm is not None and hm.enabled:
        show_head = True  # a model implies you want the head drawn
        head_model_uri_js = '"' + _resolve_glb_data_uri(hm.glb) + '"'
        head_model_size = float(hm.size_scale)
        head_model_min_px = float(hm.size_min_pixels)
        head_model_max_px = "null" if hm.size_max_pixels is None else str(float(hm.size_max_pixels))
        head_face_heading = "true" if hm.face_heading else "false"
        head_yaw_offset = float(hm.yaw_offset)
        head_pitch = float(hm.model_pitch)
        head_roll = float(hm.model_roll)
        head_smooth_samples = int(hm.smooth_samples)
        head_terrain_pitch = "true" if hm.terrain_pitch else "false"
        head_terrain_pitch_scale = float(hm.terrain_pitch_scale)
        head_min_move_m = float(hm.min_move_m)
        head_lighting_js = '"pbr"' if hm.pbr_lighting else '"flat"'
        head_tint_js = "null" if hm.tint is None else _rgb(hm.tint)
        head_use_track_color = "true" if hm.use_track_color else "false"
    else:
        head_model_uri_js = "null"
        head_model_size = 50.0
        head_model_min_px = 12.0
        head_model_max_px = "null"
        head_face_heading = "true"
        head_yaw_offset = 0.0
        head_pitch = 0.0
        head_roll = 0.0
        head_smooth_samples = 2
        head_terrain_pitch = "false"
        head_terrain_pitch_scale = 1.0
        head_min_move_m = 3.0
        head_lighting_js = '"pbr"'
        head_tint_js = "null"
        head_use_track_color = "true"

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
// === Comet over a (white) historic trail, plus a current-position head marker ===
let currentTime    = __CURRENT_TIME__;
const maxTime      = __SPAN__;
const cometTrail   = __COMET_TRAIL__;
const historyTrail = __HISTORY_TRAIL__;
let animationSpeed = __ANIMATION_SPEED__;
let isPlaying      = true;
let lastFrameTime  = 0;
let prevTime       = 0;
const fpsInterval  = 1000 / __FPS_LIMIT__;

// Historic-track config
const showHistory   = __SHOW_HISTORY__;
const historyColor  = __HISTORY_COLOR__;
const historyOpacity= __HISTORY_OPACITY__;
const fadeHistory   = __FADE_HISTORY__;

// Head-marker config
const showHead          = __SHOW_HEAD__;
const headRadius        = __HEAD_RADIUS__;
const headColorOverride = __HEAD_COLOR__;  // null -> per-subject colour
const headOutlineColor  = __HEAD_OUTLINE__;
const headOutlineWidth  = __HEAD_OUTLINE_WIDTH__;

// 3D head-model config (null modelUri -> keep the flat scatter dot)
const headModelUri      = __HEAD_MODEL_URI__;
const headModelSize     = __HEAD_MODEL_SIZE__;
const headModelMinPx    = __HEAD_MODEL_MIN_PX__;
const headModelMaxPx    = __HEAD_MODEL_MAX_PX__;
const headFaceHeading   = __HEAD_FACE_HEADING__;
const headYawOffset     = __HEAD_YAW_OFFSET__;
const headPitch         = __HEAD_PITCH__;
const headRoll          = __HEAD_ROLL__;
const headSmoothSamples = __HEAD_SMOOTH_SAMPLES__;
const headTerrainPitch  = __HEAD_TERRAIN_PITCH__;
const headTerrainScale  = __HEAD_TERRAIN_SCALE__;
const headMinMoveM      = __HEAD_MIN_MOVE_M__;   // below this window movement -> hold heading, level out
const headLighting      = __HEAD_LIGHTING__;
const headTint          = __HEAD_TINT__;
const headUseTrackColor = __HEAD_USE_TRACK_COLOR__;
const headModelEnabled  = (headModelUri != null);

// Camera rotation
const autoRotateSpeed = __AUTO_ROTATE_SPEED__;  // deg/s; 0 = off

let baseLayers = null;
let tripsBase  = null;
let tripsId    = null;
let ScatterCtor = null;     // resolved lazily
let ScenegraphCtor = null;  // resolved lazily (only when a 3D head model is configured)
let headCursors = null;     // per-feature search cursor (monotonic-time fast path)
let headLastHeading = null; // per-feature last good heading (held when stationary)
let rotateBearing  = 0;    // accumulated bearing for smooth rotation
let lastRotateTs   = 0;    // last frame timestamp used for rotation delta
let _deckViewState = null; // mirrors interactive view state so rotation + user pan compose

// Resolve a deck.gl layer constructor without hard-coding the global namespace.
function resolveLayer(name) {
  const cands = [window.deck, window.deckgl, window.DeckGL, window];
  for (const ns of cands) {
    if (ns && typeof ns[name] === 'function') return ns[name];
  }
  for (const k in window) {                       // last resort: scan
    try {
      const v = window[k];
      if (v && typeof v[name] === 'function') return v[name];
    } catch (e) { /* ignore cross-origin / getter throws */ }
  }
  return null;
}

// Append a <script> and resolve once it loads (used to pull mesh-layers + gltf loader).
function loadScript(src) {
  return new Promise(function (resolve, reject) {
    const s = document.createElement('script');
    s.src = src; s.onload = resolve; s.onerror = reject;
    document.head.appendChild(s);
  });
}

// Best-effort: make ScenegraphLayer + the glTF loader available, matching the running
// deck.gl version. Returns true if the constructor is resolvable afterwards.
async function ensureScenegraph() {
  if (resolveLayer('ScenegraphLayer')) return true;
  try {
    const ver = (window.deck && window.deck.VERSION) ? window.deck.VERSION : 'latest';
    await loadScript('https://unpkg.com/@deck.gl/mesh-layers@' + ver + '/dist.min.js');
    await loadScript('https://unpkg.com/@loaders.gl/gltf@^4.0.0/dist/dist.min.js');
    const reg = (window.deck && window.deck.registerLoaders) ||
                (window.loaders && window.loaders.registerLoaders);
    const GLTFLoader = window.loaders && window.loaders.GLTFLoader;
    if (reg && GLTFLoader) { try { reg([GLTFLoader]); } catch (e) {} }
    return !!resolveLayer('ScenegraphLayer');
  } catch (e) {
    console.warn('[draw_animated_map] could not load ScenegraphLayer/glTF loader; ' +
                 'falling back to the 2D head dot.', e);
    return false;
  }
}

// Windowed tangent around the cursor: returns a SMOOTHED heading (deg, 0 = north)
// and the terrain slope as a pitch (deg, + = uphill in the direction of travel),
// taken from the elevation (z) already baked into each [lon,lat,z] coordinate.
// k = +/- fixes to span; larger k => smoother, laggier. k = 0 => single segment.
function headTangent(C, j, last, k) {
  const ia = Math.max(0, j - k);
  const ib = Math.min(last, j + 1 + k);
  const a = C[ia], b = C[ib];
  if (!a || !b) return { heading: 0, pitch: 0, horiz: 0 };
  const midLat = (a[1] + b[1]) * 0.5 * Math.PI / 180;
  const dEastDeg  = (b[0] - a[0]) * Math.cos(midLat);
  const dNorthDeg = (b[1] - a[1]);
  const heading = (dEastDeg === 0 && dNorthDeg === 0)
    ? 0 : Math.atan2(dEastDeg, dNorthDeg) * 180 / Math.PI;
  const M_PER_DEG = 111320;
  const horiz = Math.hypot(dEastDeg, dNorthDeg) * M_PER_DEG;   // metres on the ground
  const dz = (b[2] || 0) - (a[2] || 0);                        // metres of climb/descent
  let pitch = (horiz > 0) ? Math.atan2(dz, horiz) * 180 / Math.PI : 0;
  const CAP = 20;                                              // never tip past +/-20deg
  if (pitch >  CAP) pitch =  CAP;
  if (pitch < -CAP) pitch = -CAP;
  return { heading: heading, pitch: pitch, horiz: horiz };
}

// Interpolate one feature's [lon,lat,z] at time t, using a monotonic cursor.
// When the subject is essentially stationary (tiny movement over the window) we
// HOLD the last good heading and zero the pitch, so the model doesn't spin or
// tip over while milling in place (e.g. crop-raiding).
function headAt(feature, i, t) {
  const C = feature.geometry && feature.geometry.coordinates;
  const T = feature.timestamps;
  if (!C || !T || C.length === 0) return null;
  const last = T.length - 1;
  const k = headSmoothSamples;
  function orient(g) {
    if (g.horiz >= headMinMoveM) { headLastHeading[i] = g.heading; return { heading: g.heading, pitch: g.pitch }; }
    return { heading: headLastHeading[i] || 0, pitch: 0 };   // stationary: hold + level
  }
  if (t <= T[0])    { const o = orient(headTangent(C, 0, last, k));        return { pos: C[0],    heading: o.heading, pitch: o.pitch }; }
  if (t >= T[last]) { const o = orient(headTangent(C, last - 1, last, k)); return { pos: C[last], heading: o.heading, pitch: o.pitch }; }
  let j = headCursors[i] || 0;
  if (t < T[j]) j = 0;                 // scrubbed backwards
  while (j < last && T[j + 1] < t) j++;
  headCursors[i] = j;
  const t0 = T[j], t1 = T[j + 1];
  const f = (t1 > t0) ? (t - t0) / (t1 - t0) : 0;
  const a = C[j], b = C[j + 1];
  const az = a[2] || 0, bz = b[2] || 0;
  const pos = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, az + (bz - az) * f];
  const o = orient(headTangent(C, j, last, k));
  return { pos: pos, heading: o.heading, pitch: o.pitch };
}

function headData(t) {
  const feats = tripsBase.props.data || [];
  const out = [];
  for (let i = 0; i < feats.length; i++) {
    const r = headAt(feats[i], i, t);
    if (!r) continue;
    const col = headColorOverride || (feats[i].color ? feats[i].color.slice(0, 3) : [255, 0, 0]);
    out.push({ position: r.pos, color: col, heading: r.heading, pitch: r.pitch });
  }
  return out;
}

function buildLayers() {
  const comet = tripsBase.clone({
    id: tripsId + '-comet',
    currentTime: currentTime,
    trailLength: cometTrail,
    fadeTrail: true,
    opacity: 0.98,
  });

  let history = null;
  if (showHistory) {
    // Inherit the comet's width from the TripsLayer (do NOT override getWidth).
    history = tripsBase.clone({
      id: tripsId + '-history',
      currentTime: currentTime,
      trailLength: historyTrail,     // >= span -> solid back to the start
      fadeTrail: fadeHistory,
      opacity: historyOpacity,
      getColor: historyColor,        // constant -> the whole track is this colour
      updateTriggers: { getColor: 'history' },
    });
  }

  const out = [];
  baseLayers.forEach(function (l) {
    if (l.id === tripsId) {
      if (history) out.push(history);  // bottom
      out.push(comet);                 // middle
    } else {
      out.push(l);
    }
  });

  if (showHead) {                      // head marker on top of everything
    if (headModelEnabled && ScenegraphCtor) {
      out.push(new ScenegraphCtor({
        id: tripsId + '-head',
        data: headData(currentTime),
        scenegraph: headModelUri,
        getPosition: function (d) { return d.position; },
        getOrientation: function (d) {
          const yaw = (headFaceHeading ? -d.heading : 0) + headYawOffset;
          const pitch = headPitch + (headTerrainPitch ? headTerrainScale * d.pitch : 0);
          return [pitch, yaw, headRoll];   // [pitch, yaw, roll] degrees
        },
        getColor: function (d) {
          return (headUseTrackColor && d.color) ? d.color : (headTint || [255, 255, 255]);
        },
        sizeScale: headModelSize,
        sizeMinPixels: headModelMinPx,
        sizeMaxPixels: (headModelMaxPx == null) ? Number.MAX_SAFE_INTEGER : headModelMaxPx,
        _lighting: headLighting,
        pickable: false,
        parameters: { depthTest: true },
        updateTriggers: {
          getPosition: currentTime,
          getOrientation: currentTime,
          getColor: 'head',
        },
      }));
    } else if (ScatterCtor) {          // 2D fallback dot
      out.push(new ScatterCtor({
        id: tripsId + '-head',
        data: headData(currentTime),
        getPosition: function (d) { return d.position; },
        getFillColor: function (d) { return d.color; },
        getRadius: headRadius,
        radiusUnits: 'pixels',
        stroked: headOutlineWidth > 0,
        getLineColor: headOutlineColor,
        getLineWidth: headOutlineWidth,
        lineWidthUnits: 'pixels',
        billboard: true,
        pickable: false,
        parameters: { depthTest: false },  // always visible over the terrain
        updateTriggers: { getPosition: currentTime, getFillColor: 'head' },
      }));
    }
  }
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

  prevTime = currentTime;
  if (currentTime < maxTime) {
    currentTime = Math.min(maxTime, currentTime + animationSpeed);
  } else {
    isPlaying = false;        // Stop animation at the end
  }

  window.deckInstance.setProps({ layers: buildLayers() });

  if (autoRotateSpeed !== 0 && _deckViewState) {
    const dt = lastRotateTs ? (timestamp - lastRotateTs) : 0;
    lastRotateTs = timestamp;
    rotateBearing = (rotateBearing + autoRotateSpeed * dt / 1000) % 360;
    window.deckInstance.setProps({ viewState: Object.assign({}, _deckViewState, { bearing: rotateBearing }) });
  }

  requestAnimationFrame(frame);
}

// Start
let __headReady = !headModelEnabled;   // if no model, head is "ready" immediately
const __startWhenReady = setInterval(function () {
  if (window.deckInstance && window.deckInstance.props && window.deckInstance.props.layers) {
    clearInterval(__startWhenReady);

    baseLayers  = window.deckInstance.props.layers;
    tripsBase   = baseLayers.find(l => 'currentTime' in l.props);
    tripsId     = tripsBase.id;
    headCursors = new Array((tripsBase.props.data || []).length).fill(0);
    headLastHeading = new Array((tripsBase.props.data || []).length).fill(0);

    // Seed rotation from the initial view state and hook onViewStateChange so
    // user panning/zooming is preserved while rotation is applied.
    if (autoRotateSpeed !== 0) {
      const ivs = window.deckInstance.props && window.deckInstance.props.initialViewState;
      if (ivs) {
        _deckViewState = Object.assign({}, ivs);
        rotateBearing  = _deckViewState.bearing || 0;
      }
      const _origOnVS = window.deckInstance.props.onViewStateChange;
      window.deckInstance.setProps({
        onViewStateChange: function (params) {
          _deckViewState = params.viewState;
          if (_origOnVS) _origOnVS(params);
        }
      });
    }

    if (showHead) {
      ScatterCtor = resolveLayer('ScatterplotLayer');
      if (!ScatterCtor) {
        console.warn('[draw_animated_map] ScatterplotLayer constructor not found; ' +
                     '2D head marker disabled. Trail animation is unaffected.');
      }
      if (headModelEnabled) {
        // Load mesh-layers + glTF loader, then enable the 3D head on the next frame.
        ensureScenegraph().then(function (ok) {
          if (ok) ScenegraphCtor = resolveLayer('ScenegraphLayer');
          __headReady = true;
          if (window.deckInstance) window.deckInstance.setProps({ layers: buildLayers() });
        });
      }
    }

    requestAnimationFrame(frame);
  }
}, 200);

// --- Deterministic render bridge (used by the server-side MP4 exporter) ---------
// Lets a headless driver pause autoplay and paint an exact frame at time t.
window.__tripsAnim = {
  get ready()    { return !!(window.deckInstance && tripsBase); },
  get headReady() { return __headReady; },
  get span()     { return maxTime; },
  get speed()    { return animationSpeed; },
  // Natural playback length (s): maxTime / per-tick advance, at ~60 rAF ticks/s.
  get durationSec() { return animationSpeed > 0 ? (maxTime / animationSpeed) / 60 : 0; },
  pause() { isPlaying = false; },
  play()  { if (!isPlaying) { isPlaying = true; requestAnimationFrame(frame); } },
  renderAt(t) {
    isPlaying = false;
    currentTime = Math.max(0, Math.min(maxTime, t));
    if (window.deckInstance) window.deckInstance.setProps({ layers: buildLayers() });
  }
};
</script>
"""
    animation_js = (
        animation_js.replace("__CURRENT_TIME__", str(current_time))
        .replace("__SPAN__", str(span))
        .replace("__COMET_TRAIL__", str(comet_trail))
        .replace("__HISTORY_TRAIL__", str(history_trail))
        .replace("__ANIMATION_SPEED__", str(animation_speed))
        .replace("__FPS_LIMIT__", str(fps_limit))
        .replace("__SHOW_HISTORY__", "true" if show_history else "false")
        .replace("__HISTORY_COLOR__", history_color_js)
        .replace("__HISTORY_OPACITY__", str(history_opacity))
        .replace("__FADE_HISTORY__", fade_history_js)
        .replace("__SHOW_HEAD__", "true" if show_head else "false")
        .replace("__HEAD_RADIUS__", str(head_radius))
        .replace("__HEAD_COLOR__", head_color_js)
        .replace("__HEAD_OUTLINE__", head_outline_js)
        .replace("__HEAD_OUTLINE_WIDTH__", str(head_outline_width))
        .replace("__HEAD_MODEL_URI__", head_model_uri_js)
        .replace("__HEAD_MODEL_SIZE__", str(head_model_size))
        .replace("__HEAD_MODEL_MIN_PX__", str(head_model_min_px))
        .replace("__HEAD_MODEL_MAX_PX__", head_model_max_px)
        .replace("__HEAD_FACE_HEADING__", head_face_heading)
        .replace("__HEAD_YAW_OFFSET__", str(head_yaw_offset))
        .replace("__HEAD_PITCH__", str(head_pitch))
        .replace("__HEAD_ROLL__", str(head_roll))
        .replace("__HEAD_SMOOTH_SAMPLES__", str(head_smooth_samples))
        .replace("__HEAD_TERRAIN_PITCH__", head_terrain_pitch)
        .replace("__HEAD_TERRAIN_SCALE__", str(head_terrain_pitch_scale))
        .replace("__HEAD_MIN_MOVE_M__", str(head_min_move_m))
        .replace("__HEAD_LIGHTING__", head_lighting_js)
        .replace("__HEAD_TINT__", head_tint_js)
        .replace("__HEAD_USE_TRACK_COLOR__", head_use_track_color)
        .replace("__AUTO_ROTATE_SPEED__", str(auto_rotate_speed))
    )
    html_str = html_str.replace("</body>", animation_js + "</body>")
    return html_str


@task
def create_timeline_animation(
    animation_speed: Annotated[float, AdvancedField(default=1000, ge=0)] = 1000,
    fade_ratio: Annotated[float, AdvancedField(default=0.95, gt=0, le=1)] = 0.95,
    fps_limit: Annotated[float, AdvancedField(default=30.0, gt=0)] = 30.0,
    show_history: Annotated[bool, AdvancedField(default=True)] = True,
    history_color: Annotated[
        tuple[int, int, int], AdvancedField(default=(255, 255, 255), json_schema_extra={"items": {"type": "integer"}})
    ] = (255, 255, 255),
    history_opacity: Annotated[float, AdvancedField(default=0.85, ge=0, le=1)] = 0.85,
    fade_history: Annotated[bool, AdvancedField(default=False)] = False,
    show_head: Annotated[bool, AdvancedField(default=True)] = True,
    head_radius: Annotated[float, AdvancedField(default=6.0, gt=0)] = 6.0,
    head_color: Annotated[
        tuple[int, int, int] | None, AdvancedField(default=None, json_schema_extra={"items": {"type": "integer"}})
    ] = None,
    head_outline_color: Annotated[
        tuple[int, int, int], AdvancedField(default=(255, 255, 255), json_schema_extra={"items": {"type": "integer"}})
    ] = (255, 255, 255),
    head_outline_width: Annotated[float, AdvancedField(default=1.5, ge=0)] = 1.5,
    auto_rotate_speed: Annotated[
        float,
        AdvancedField(
            default=0.0,
            description="Camera rotation speed in degrees per second. "
            "0 = off; positive = clockwise; negative = counter-clockwise.",
        ),
    ] = 0.0,
) -> Annotated[TimelineAnimation, Field()]:
    """Construct a TimelineAnimation config, exposing animation_speed as the primary form field."""
    return TimelineAnimation(
        animation_speed=animation_speed,
        fade_ratio=fade_ratio,
        fps_limit=fps_limit,
        show_history=show_history,
        history_color=history_color,
        history_opacity=history_opacity,
        fade_history=fade_history,
        show_head=show_head,
        head_radius=head_radius,
        head_color=head_color,
        head_outline_color=head_outline_color,
        head_outline_width=head_outline_width,
        auto_rotate_speed=auto_rotate_speed,
    )


@task
def create_elevation_decoder(
    exaggeration: Annotated[
        float,
        AdvancedField(
            default=1.0, gt=0, description="Vertical exaggeration factor. 1.0 = true scale, 2.0 = 2x heights."
        ),
    ] = 1.0,
    r_scaler: Annotated[float, AdvancedField(default=256.0)] = 256.0,
    g_scaler: Annotated[float, AdvancedField(default=1.0)] = 1.0,
    b_scaler: Annotated[float, AdvancedField(default=1 / 256)] = 1 / 256,
    offset: Annotated[float, AdvancedField(default=-32768.0)] = -32768.0,
) -> Annotated[dict, Field()]:
    """Build an RGB->elevation decoder with vertical exaggeration baked in.

    deck.gl's TerrainLayer has no elevation-scale prop, so exaggeration is applied by
    scaling every decoder term by `exaggeration`. Feed the result into BOTH
    create_terrain_layer and create_terrain_sampling so the mesh and the trips agree.
    Defaults are Terrarium.
    """
    return {
        "rScaler": r_scaler * exaggeration,
        "gScaler": g_scaler * exaggeration,
        "bScaler": b_scaler * exaggeration,
        "offset": offset * exaggeration,
    }


def _zoom_from_bbox(minx, miny, maxx, maxy, map_width_px=800, map_height_px=600) -> float:
    """
    Calculate zoom level to fit bounding box in a given map size.
    This approach considers both dimensions for optimal fitting.

    Args:
        minx, miny, maxx, maxy (float): bounding box coordinates, must be in EPSG:4326
        map_width_px, map_height_px (int): target map dimensions in pixels

    Returns:
        float: zoom level that fits the bbox in the map
    """
    width_deg = abs(maxx - minx)
    height_deg = abs(maxy - miny)
    center_lat = (miny + maxy) / 2

    # Convert to km
    height_km = height_deg * 111.0
    width_km = width_deg * 111.0 * abs(math.cos(math.radians(center_lat)))

    world_width_km = 40075
    world_height_km = 40075

    zoom_for_width = math.log2(world_width_km * map_width_px / (512 * width_km))
    zoom_for_height = math.log2(world_height_km * map_height_px / (512 * height_km))

    zoom = min(zoom_for_width, zoom_for_height)
    zoom = round(max(0, min(20, zoom)), 2)
    return zoom


@task
def compute_view_state_from_gdf(
    gdf,
    pitch: Annotated[
        int,
        AdvancedField(
            default=45,
            ge=0,
            le=90,
            description="Camera tilt in degrees (0 = top-down, 90 = horizon). "
            "45° gives a natural 3-D perspective over terrain.",
        ),
    ] = 45,
    bearing: Annotated[
        int,
        AdvancedField(
            default=0,
            ge=-180,
            le=180,
            description="Camera compass heading in degrees (0 = north, 90 = east, -90 = west). "
            "Rotates the map so a different cardinal direction faces up.",
        ),
    ] = 0,
) -> ViewState:
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")

    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0
    zoom = _zoom_from_bbox(minx, miny, maxx, maxy)
    return ViewState(longitude=center_lon, latitude=center_lat, zoom=zoom, pitch=pitch, bearing=bearing)

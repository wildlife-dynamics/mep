import os
import io
import json
import base64
import pathlib
import logging
import requests
import shapely
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import median_filter
import geopandas as gpd
import concurrent.futures as cf
from dataclasses import dataclass, fields, replace
from pydantic import BaseModel, Field, ConfigDict, model_validator
from shapely.geometry import LineString
from typing import Annotated, Literal, cast
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from ecoscope.platform.schemas import TrajectoryGDF
from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
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
# Matches create_terrain_layer's default max_zoom so sampled z follows the rendered mesh.
ELEVATION_SAMPLE_ZOOM = 15
# Decoded-units jump from the 3x3 median treated as a DEM artifact (see _despike_dem).
DEM_SPIKE_THRESHOLD = 150.0
DEFAULT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRARIUM_ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}
DEFAULT_TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
SURFACE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

TerrainStrategy = Annotated[Literal["best-available", "no-overlap", "never"], PydeckAnnotation]


class _BasemapFields(BaseModel):
    """Shared fields available on every basemap option (Default and Custom alike)."""

    elevation_decoder: Annotated[
        dict | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="RGB->elevation decoder. None -> Terrarium default. Feed "
            "create_elevation_decoder's output here to apply vertical exaggeration; the same "
            "decoder is used both to render the terrain mesh and to sample trip elevations.",
        ),
    ] = None


class DefaultBasemap(_BasemapFields):
    """Standard basemap: AWS Terrarium elevation tiles draped with ArcGIS World Imagery."""

    model_config = ConfigDict(json_schema_extra={"title": "Default"})
    preset: Annotated[Literal["default"], Field(default="default", title="Basemap")] = "default"


class CustomBasemap(_BasemapFields):
    """Provide your own elevation and/or texture tile URL templates."""

    model_config = ConfigDict(json_schema_extra={"title": "Custom"})
    preset: Annotated[Literal["custom"], Field(default="custom", title="Basemap")] = "custom"
    tile_urls: Annotated[
        dict | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Custom elevation/texture tile URLs. None -> defaults. Feed "
            "set_custom_basemap_urls's output here.",
        ),
    ] = None


BasemapOption = Annotated[DefaultBasemap | CustomBasemap, Field(discriminator="preset")]


class DotMarker(BaseModel):
    """Flat circle at each subject's current position (deck.gl ScatterplotLayer).

    Field names mirror PointLayerStyle. Colours are constants rather than column accessors
    because the marker's data is rebuilt in the browser every frame.
    """

    model_config = ConfigDict(json_schema_extra={"title": "Dot"})
    marker: Annotated[Literal["dot"], Field(default="dot", title="Marker icon")] = "dot"
    get_radius: Annotated[float, AdvancedField(default=6.0, gt=0)] = 6.0
    radius_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    get_fill_color: Annotated[
        tuple[int, int, int] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="RGB fill. None -> each subject's own track colour.",
            json_schema_extra={"items": {"type": "integer"}},
        ),
    ] = None
    stroked: Annotated[bool, AdvancedField(default=True)] = True
    get_line_color: Annotated[
        tuple[int, int, int], AdvancedField(default=(255, 255, 255), json_schema_extra={"items": {"type": "integer"}})
    ] = (255, 255, 255)
    get_line_width: Annotated[float, AdvancedField(default=1.5, ge=0)] = 1.5
    line_width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    opacity: Annotated[float, AdvancedField(default=1, ge=0, le=1)] = 1


class NoMarker(BaseModel):
    """No marker at the current position; only the trails are drawn."""

    model_config = ConfigDict(json_schema_extra={"title": "None"})
    marker: Annotated[Literal["none"], Field(default="none", title="Marker icon")] = "none"


class ScenegraphLayerDefinition(BaseModel):
    """An animated 3D head built from a glTF/GLB model (deck.gl ScenegraphLayer).

    Create one with create_scenegraph_layer() and use it as
    TripsAnimation.head. Its position and heading are driven per-frame
    from the TripsLayer, so it follows each subject's current location. `glb` accepts
    an http(s) URL, a `data:` URI, or a local file path (read and embedded as a data
    URI). None -> the preset elephant model. If the ScenegraphLayer
    constructor or the glTF loader can't be resolved at runtime, the head silently
    falls back to the flat ScatterplotLayer dot. For a ready-tuned animal, use
    AnimalModelMarker instead.
    See https://deck.gl/docs/api-reference/mesh-layers/scenegraph-layer for more info.
    """

    model_config = ConfigDict(json_schema_extra={"title": "3D model (custom)"}, protected_namespaces=())
    marker: Annotated[Literal["model"], Field(default="model", title="Marker icon")] = "model"
    glb: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            title="3D model (GLB)",
            description="GLB source: an http(s) URL, a data: URI, or a local file path. "
            "None -> the preset elephant model.",
        ),
    ] = None
    size_scale: Annotated[
        float,
        AdvancedField(default=50.0, gt=0, title="Size scale", description="Model size multiplier. Tune to your scene."),
    ] = 50.0
    size_min_pixels: Annotated[
        float,
        AdvancedField(
            default=12.0,
            ge=1,
            le=200,
            title="Min size (px)",
            description="Clamp the on-screen model to at least this many pixels so it stays visible when zoomed out.",
        ),
    ] = 12.0
    size_max_pixels: Annotated[
        Annotated[float, Field(ge=1, le=500)] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            title="Max size (px)",
            description="Upper clamp on the model's on-screen size in pixels. Must be at least Min size.",
        ),
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
        AdvancedField(
            default=True,
            title="Realistic lighting (PBR)",
            description="Shade the model with physically-based lighting. Uncheck for flat shading, "
            "which shows subject colors more accurately.",
        ),
    ] = True
    tint: Annotated[
        list[int] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            title="Tint",
            description="Optional RGB tint over the model as [R, G, B]. None -> the model's own materials.",
        ),
    ] = [220, 220, 255]
    use_track_color: Annotated[
        bool,
        AdvancedField(
            default=True,
            title="Use subject color",
            description="Colour the model with each subject's track colour. "
            "False -> use `tint` (or the model's own materials). Note: the colour "
            "multiplies the model's material, so it reads truest with a light/neutral "
            "glb and flat lighting (pbr_lighting=False).",
        ),
    ] = True

    @model_validator(mode="after")
    def _check_size_range(self):
        if self.size_max_pixels is not None and self.size_max_pixels < self.size_min_pixels:
            raise ValueError(
                f"Max size ({self.size_max_pixels} px) must be at least Min size ({self.size_min_pixels} px)."
            )
        return self


_ANIMAL_MODEL_URL = "https://raw.githubusercontent.com/wildlife-dynamics/animate_subject_tracks/main/models/{}.glb"
# Tuned per model in custom-tripslayer/test_trips_layer.ipynb. Shared by all of them: roll 90 to
# stand the models upright, heading smoothing and an 8 m move threshold against GPS jitter, level
# on slopes, PBR shading, neutral tint, coloured by track.
_ANIMAL_MODEL_COMMON = dict(
    face_heading=True,
    model_pitch=0.0,
    model_roll=90.0,
    smooth_samples=8,
    terrain_pitch=False,
    min_move_m=8.0,
    tint=[255, 255, 255],
)
_ANIMAL_MODELS = {  # size_scale, size_min_pixels, size_max_pixels, yaw_offset (the nose direction differs)
    "elephant": (100.0, 15.0, 120.0, 0.0),
    "giraffe": (75.0, 25.0, 120.0, 0.0),
    "cheetah": (75.0, 25.0, 120.0, 90.0),
    "leopard": (75.0, 25.0, 120.0, 180.0),
    "lion": (1.0, 1.0, 10.0, 180.0),
}
AnimalModel = Literal["elephant", "giraffe", "cheetah", "leopard", "lion"]


class AnimalModelMarker(BaseModel):
    """A ready-tuned 3D animal model: pick the animal, everything else is preset.

    For your own GLB, or to tune orientation/size by hand, use the custom 3D model
    (ScenegraphLayerDefinition) instead.
    """

    model_config = ConfigDict(json_schema_extra={"title": "3D animal"})
    marker: Annotated[Literal["animal"], Field(default="animal", title="Marker icon")] = "animal"
    animal: Annotated[AnimalModel, Field(default="elephant", description="Which preset model to show.")] = "elephant"
    size: Annotated[
        float,
        AdvancedField(default=1.0, gt=0, description="Multiplier on the preset size (2 = twice as big)."),
    ] = 1.0
    use_track_color: Annotated[
        bool, AdvancedField(default=True, title="Use subject color", description="Colour by each subject's track.")
    ] = True
    pbr_lighting: Annotated[
        bool,
        AdvancedField(
            default=True, title="Realistic lighting (PBR)", description="Uncheck for flat shading (truer colours)."
        ),
    ] = True

    def to_model(self) -> "ScenegraphLayerDefinition":
        scale, min_px, max_px, yaw = _ANIMAL_MODELS[self.animal]
        return ScenegraphLayerDefinition(
            glb=_ANIMAL_MODEL_URL.format(self.animal),
            size_scale=scale * self.size,
            size_min_pixels=min_px * self.size,
            size_max_pixels=max_px * self.size,
            yaw_offset=yaw,
            use_track_color=self.use_track_color,
            pbr_lighting=self.pbr_lighting,
            **_ANIMAL_MODEL_COMMON,
        )


# Bare union: draw_animated_map sets the discriminator on its own AdvancedField (the compiler only reads
# the first FieldInfo in an Annotated, so an inner Field here would drop the title and advanced flag).
HeadMarker = DotMarker | AnimalModelMarker | ScenegraphLayerDefinition | NoMarker


def _resolve_glb_data_uri(glb: str | None) -> str:
    """Resolve a ScenegraphLayerDefinition.glb source to something deck.gl can load.

    URLs and data: URIs pass through; a local path is read and base64-embedded so the
    output HTML stays self-contained; None -> the preset elephant model.
    """
    if glb is None:
        return _ANIMAL_MODEL_URL.format("elephant")
    if glb.startswith(("http://", "https://", "data:")):
        return glb
    raw = pathlib.Path(glb).read_bytes()
    return "data:model/gltf-binary;base64," + base64.b64encode(raw).decode()


class PlaybackControls(BaseModel):
    """The playback bar drawn on an animated map. Every part can be switched off."""

    visible: bool = Field(default=True, description="Show the playback bar at all.")
    show_play: bool = Field(default=True, description="Play/pause button.")
    show_restart: bool = Field(default=True, description="Restart button.")
    show_scrubber: bool = Field(default=True, description="Slider for jumping to any moment.")
    show_clock: bool = Field(default=True, description="Playback position / total length, e.g. 0:12 / 0:30.")
    show_time: bool = Field(default=True, description="Current time in the data.")
    time_format: Literal["datetime", "date", "elapsed"] = Field(
        default="datetime",
        description="How the data time is shown: 'datetime' (2024-01-01 06:00 UTC), 'date' (2024-01-01), "
        "or 'elapsed' since the start (6.0 h). Non-date data always shows elapsed.",
    )
    show_speed: bool = Field(default=True, description="Button cycling through `speeds`.")
    speeds: list[Annotated[float, Field(gt=0)]] = Field(
        default=[0.5, 1, 2, 4],
        min_length=1,
        description="Speed multipliers the speed button cycles through. Playback starts at 1x.",
    )
    position: Literal["bottom", "top"] = Field(default="bottom", description="Where the bar sits on the map.")


class TimelineAnimation(BaseModel):
    """The shared clock for draw_animated_map. Knows nothing about individual layers;
    each animated layer carries its own TripsAnimation / TimeWindowAnimation."""

    duration_s: float = Field(
        default=30.0,
        gt=0,
        description="Playback length in seconds, from the start of the timeline to its end.",
    )
    fps_limit: float = Field(
        default=30.0,
        gt=0,
        description="Maximum animation frames per second.",
    )
    controls: PlaybackControls = Field(
        default_factory=PlaybackControls,
        description="The playback bar.",
    )
    auto_rotate_speed: Annotated[
        float,
        AdvancedField(
            default=0.0,
            description="Camera rotation speed in degrees per second while the animation plays. "
            "0 = off; positive = clockwise; negative = counter-clockwise.",
        ),
    ] = 0.0


class TripsAnimation(BaseModel):
    """Animates a TripsLayer: a coloured comet over a historic trail, plus a head marker."""

    model_config = ConfigDict(json_schema_extra={"title": "Trips"})
    kind: Annotated[Literal["trips"], Field(default="trips", title="Animation")] = "trips"
    comet_ratio: float = Field(
        default=0.3,
        gt=0,
        le=1,
        description="Comet-tail length as a fraction of the total time span (0–1].",
    )
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
    head: Annotated[
        HeadMarker,
        AdvancedField(
            default=DotMarker(),
            discriminator="marker",
            title="Marker icon",
            description="Marker drawn at each subject's current position: a flat dot, a preset 3D animal, "
            "your own 3D glTF/GLB model, or none.",
        ),
    ] = DotMarker()


class TimeWindowAnimation(BaseModel):
    """Animates any layer with one time per row (points, paths, polygons, text, ...):
    only rows whose time falls inside the window ending at the current time are drawn."""

    model_config = ConfigDict(json_schema_extra={"title": "Time window"})
    kind: Annotated[Literal["window"], Field(default="window", title="Animation")] = "window"
    time_col: str = Field(description="Column holding each row's time (datetime or epoch seconds).")
    window_s: Annotated[float, Field(gt=0)] | SkipJsonSchema[None] = Field(
        default=None,
        description="Seconds of data visible behind the current time. None -> everything up to now.",
    )
    fade_s: float = Field(
        default=0.0,
        ge=0,
        description="Seconds over which rows fade out before leaving the window. 0 -> hard cut-off.",
    )


# Bare union for the same reason as HeadMarker: the task field sets the discriminator.
LayerAnimation = TripsAnimation | TimeWindowAnimation


@dataclass
class AnimatedLayerDefinition(LayerDefinition):
    """A LayerDefinition plus the animation draw_animated_map drives it with."""

    animation: LayerAnimation | None = None


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

    Consumed by drape_trips_on_terrain; skip that step for flat 2D paths.
    """

    offset: float = Field(default=30.0, description="Metres added above the sampled ground at every vertex.")
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


@register()
def create_terrain_sampling(
    elevation_data: Annotated[
        str, Field(description="Elevation tile URL template. Must match the TerrainLayer's elevation_data.")
    ] = DEFAULT_TERRAIN_URL,
    elevation_decoder: Annotated[
        dict | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="RGB->elevation decoder. Must match the TerrainLayer's elevation_decoder. "
            "None -> Terrarium default.",
        ),
    ] = None,
    offset: Annotated[
        float, AdvancedField(default=30, description="Metres added above the sampled ground at every vertex.")
    ] = 30.0,
    ground_elevation: Annotated[
        float, AdvancedField(default=1000.0, description="Constant ground used only if DEM sampling fails.")
    ] = 1000.0,
) -> Annotated[TerrainSampling, Field()]:
    """Creates the elevation sampling config consumed by drape_trips_on_terrain."""
    return TerrainSampling(
        elevation_data=elevation_data,
        elevation_decoder=elevation_decoder or dict(TERRARIUM_ELEVATION_DECODER),
        offset=offset,
        ground_elevation=ground_elevation,
    )


@register()
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


@register()
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


@register()
def create_animal_model(
    animal: Annotated[AnimalModel, Field(description="Which preset 3D animal to show.")] = "elephant",
    size: Annotated[float, AdvancedField(default=1.0, gt=0, description="Multiplier on the preset size.")] = 1.0,
    use_track_color: Annotated[bool, AdvancedField(default=True, description="Colour by each subject's track.")] = True,
    pbr_lighting: Annotated[
        bool, AdvancedField(default=True, description="Realistic lighting; uncheck for flat shading (truer colours).")
    ] = True,
) -> Annotated[AnimalModelMarker, Field()]:
    """A ready-tuned 3D animal head marker for create_trips_animation. For your own GLB use
    create_scenegraph_layer."""
    return AnimalModelMarker(animal=animal, size=size, use_track_color=use_track_color, pbr_lighting=pbr_lighting)


@register()
def create_scenegraph_layer(
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

    Use the result as create_trips_animation's head. The model is placed at each
    subject's current position and (optionally) rotated to face its direction of travel,
    driven per-frame from the TripsLayer. With glb=None it uses the bundled default
    (elephant). If ScenegraphLayer / the glTF loader can't be resolved in the browser,
    the head falls back to the flat ScatterplotLayer dot.
    """
    return ScenegraphLayerDefinition(
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


def _layer_id(layer_def, idx: int) -> str:
    """The deck.gl id for a geo layer: its own id if it has one, else its position."""
    return getattr(layer_def, "id", None) or f"layer-{idx}"


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

    DEFAULT_WIDGETS = [  # same defaults as upstream draw_map
        pdk.Widget("NorthArrowWidget", placement="top-left", id="NorthArrowWidget", style={"transform": "scale(0.8)"}),
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

    for idx, layer_def in enumerate(geo_layers):
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
                id=_layer_id(layer_def, idx),  # stable, so the animation script can find this layer
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
        parameters={"depthTest": any(getattr(layer, "extruded", False) for layer in map_layers)},
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
    dem = arr[:, :, 0] * d["rScaler"] + arr[:, :, 1] * d["gScaler"] + arr[:, :, 2] * d["bScaler"] + d["offset"]
    return _despike_dem(dem)


def _despike_dem(dem, threshold=DEM_SPIKE_THRESHOLD):
    """Replace pixels far from their 3x3 median with that median.

    Terrarium tiles carry bad seams (e.g. a whole pixel row ~2 km too low along 1-degree
    SRTM boundaries); a vertex sampled there drops out of the terrain like a candlestick.
    A one-pixel row or column is outvoted by its neighbours; real slopes over ~3 pixels
    stay well under the threshold. `mirror` keeps a bad edge row from voting for itself.
    """
    med = median_filter(dem, size=3, mode="mirror")
    return np.where(np.abs(dem - med) > threshold, med, dem)


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


def _stitch_segments(segments: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate time-ordered segments into one (lon, lat) vertex array + epoch-second timestamps.

    Each segment's vertices are spaced evenly in time between its segment_start and segment_end.
    A segment whose first vertex repeats the previous segment's last vertex contributes it once.
    """
    coords, seg_idx = shapely.get_coordinates(segments.geometry.values, return_index=True)
    counts = np.bincount(seg_idx, minlength=len(segments))
    seg_first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(len(coords)) - seg_first[seg_idx]  # vertex position within its segment
    frac = pos / np.maximum(counts[seg_idx] - 1, 1)  # single-vertex segment -> frac 0

    start = segments["segment_start"].map(pd.Timestamp.timestamp).to_numpy(dtype=np.float64)
    end = segments["segment_end"].map(pd.Timestamp.timestamp).to_numpy(dtype=np.float64)
    times = start[seg_idx] + (end - start)[seg_idx] * frac

    dup = np.zeros(len(coords), dtype=bool)
    dup[1:] = (pos[1:] == 0) & (coords[1:] == coords[:-1]).all(axis=1)
    return coords[~dup], times[~dup]


@register()
def trajectory_to_trips(
    trajectory_gdf: TrajectoryGDF,
    groupby_col: Annotated[
        str, Field(description="Column identifying each track; one trip is built per unique value.")
    ] = "groupby_col",
    keep_cols: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(description="Extra columns to carry onto each trip (first value per group)."),
    ] = None,
) -> AnyGeoDataFrame:
    """Stitch each group's segments into one 2D lon/lat LineString with per-vertex timestamps.

    Pipe the result through `drape_trips_on_terrain` for 3D paths over a TerrainLayer.
    """
    keep_cols = [c for c in (keep_cols or []) if c != groupby_col]

    rows = []
    for key, g in trajectory_gdf.groupby(groupby_col):
        g = g.sort_values("segment_start")
        coords, times = _stitch_segments(g)
        if len(coords) < 2:  # LineString needs >= 2 vertices
            continue
        row = {groupby_col: key, **g[keep_cols].iloc[0].to_dict()}
        # Absolute epoch seconds; draw_animated_map rebases every animated layer onto one clock.
        row["timestamps"] = times.tolist()
        row["geometry"] = LineString(coords)
        rows.append(row)

    columns = [groupby_col, *keep_cols, "timestamps", "geometry"]
    trips = gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs="EPSG:4326")
    return cast(AnyGeoDataFrame, trips)


@register()
def drape_trips_on_terrain(
    trips_gdf: AnyGeoDataFrame,
    terrain: Annotated[TerrainSampling, Field(description="Elevation sampling config.")] = TerrainSampling(),
) -> AnyGeoDataFrame:
    """Set every LineString vertex's z to the sampled ground elevation + terrain.offset.

    All vertices across all rows are sampled in one batched call, so each DEM tile is fetched
    once. Falls back to a constant terrain.ground_elevation if sampling fails.
    """
    lonlats, row_idx = shapely.get_coordinates(trips_gdf.geometry.values, return_index=True)
    try:
        ground = np.asarray(
            sample_elevations(
                lonlats,
                zoom=ELEVATION_SAMPLE_ZOOM,
                url=terrain.elevation_data,
                decoder=terrain.elevation_decoder,
            )
        )
    except Exception as exc:  # network/tile failure -> safe constant fallback
        logger.warning(
            "drape_trips_on_terrain: terrain sampling failed (%s); using constant ground_elevation=%s",
            exc,
            terrain.ground_elevation,
        )
        ground = np.full(len(lonlats), terrain.ground_elevation)

    xyz = np.column_stack([lonlats, ground + terrain.offset])
    draped = shapely.linestrings(xyz, indices=row_idx)
    return cast(
        AnyGeoDataFrame, trips_gdf.set_geometry(gpd.GeoSeries(draped, index=trips_gdf.index, crs=trips_gdf.crs))
    )


def _to_epoch_seconds(values: pd.Series) -> np.ndarray:
    """Datetime (naive or tz-aware) or numeric column -> float epoch seconds."""
    if not pd.api.types.is_numeric_dtype(values):
        values = pd.to_datetime(values)
    if pd.api.types.is_datetime64_any_dtype(values):
        epoch = pd.Timestamp(0, tz=values.dt.tz)
        return (values - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
    return values.to_numpy(dtype=np.float64)


def _head_spec(head: HeadMarker) -> dict:
    """Head-marker config for the trips animator. The dot style is always included
    because it is also the fallback when a 3D model can't be loaded."""
    if isinstance(head, NoMarker):
        return {"kind": "none"}
    if isinstance(head, AnimalModelMarker):
        head = head.to_model()
    dot = head if isinstance(head, DotMarker) else DotMarker()
    spec: dict = {
        "kind": "dot",
        "color": list(dot.get_fill_color) if dot.get_fill_color is not None else None,
        "smoothSamples": 2,
        "minMoveM": 3.0,
        "dot": {
            "getRadius": dot.get_radius,
            "radiusUnits": dot.radius_units,
            "stroked": dot.stroked and dot.get_line_width > 0,
            "getLineColor": list(dot.get_line_color),
            "getLineWidth": dot.get_line_width,
            "lineWidthUnits": dot.line_width_units,
            "opacity": dot.opacity,
        },
    }
    if isinstance(head, ScenegraphLayerDefinition):
        spec.update(
            kind="model",
            color=None,
            smoothSamples=head.smooth_samples,
            minMoveM=head.min_move_m,
            model={
                "uri": _resolve_glb_data_uri(head.glb),
                "size": head.size_scale,
                "minPx": head.size_min_pixels,
                "maxPx": head.size_max_pixels,
                "faceHeading": head.face_heading,
                "yawOffset": head.yaw_offset,
                "pitch": head.model_pitch,
                "roll": head.model_roll,
                "terrainPitch": head.terrain_pitch,
                "terrainScale": head.terrain_pitch_scale,
                "lighting": "pbr" if head.pbr_lighting else "flat",
                "tint": list(head.tint) if head.tint is not None else None,
                "useTrackColor": head.use_track_color,
            },
        )
    return spec


@register()
def animate_layer(
    layer: Annotated[LayerDefinition, Field(description="The layer to animate.", exclude=True)],
    animation: Annotated[
        LayerAnimation,
        Field(
            discriminator="kind",
            description="How the layer changes over time: a trips comet (TripsLayer only) or a time "
            "window (any layer with a time column).",
        ),
    ],
) -> Annotated[LayerDefinition, Field()]:
    """Attach an animation to a layer so draw_animated_map drives it from the shared clock."""
    if isinstance(animation, TripsAnimation) and layer.layer_type != "TripsLayer":
        raise ValueError(f"TripsAnimation needs a TripsLayer, got {layer.layer_type}.")
    base = {f.name: getattr(layer, f.name) for f in fields(LayerDefinition)}
    return AnimatedLayerDefinition(**base, animation=animation)


@register()
def draw_animated_map(
    geo_layers: Annotated[
        LayerDefinition | list[LayerDefinition] | SkipJsonSchema[None],
        Field(
            description="Map layers. Layers from animate_layer are animated; a plain TripsLayer is "
            "animated with the default TripsAnimation; everything else is static.",
            exclude=True,
        ),
    ] = None,
    tile_layers: Annotated[
        list | SkipJsonSchema[None],
        Field(description="Base maps and/or overlays, as in draw_map."),
    ] = None,
    timeline: Annotated[
        TimelineAnimation | SkipJsonSchema[None],
        AdvancedField(default=TimelineAnimation(), description="Shared clock settings."),
    ] = None,
    static: Annotated[bool, Field(default=False)] = False,
    title: Annotated[str | SkipJsonSchema[None], AdvancedField(default="")] = None,
    legend_style: Annotated[LegendStyle | SkipJsonSchema[None], AdvancedField(default=LegendStyle())] = None,
    max_zoom: Annotated[int, AdvancedField(default=20)] = 20,
    view_state: Annotated[ViewState | SkipJsonSchema[None], AdvancedField(default=ViewState())] = None,
    widget_id: Annotated[str | SkipJsonSchema[None], Field(default=None, exclude=True)] = None,
) -> Annotated[str, Field()]:
    """Like draw_map, but animates layers over time from one shared clock.

    Returns a static HTML string (same contract as draw_map). Every animated layer is
    rebased onto a common timeline starting at its earliest time, so layers stay in step.
    """
    timeline = timeline or TimelineAnimation()
    geo_list = [geo_layers] if isinstance(geo_layers, LayerDefinition) else list(geo_layers or [])

    # --- Which layers animate, and the times each one covers ------------------------
    animated = []  # (index in geo_list, layer, animation, epoch-second times per row)
    for i, ld in enumerate(geo_list):
        anim = getattr(ld, "animation", None)
        if anim is None and ld.layer_type == "TripsLayer":
            anim = TripsAnimation()
        if anim is None:
            continue
        if ld.geodataframe is None:
            raise ValueError(f"Animated layer {ld.layer_type} needs a geodataframe, not a data_url.")
        if isinstance(anim, TripsAnimation):
            ts_col = getattr(ld.layer_style, "get_timestamps", "timestamps")
            times = [np.asarray(ts, dtype=np.float64) for ts in ld.geodataframe[ts_col]]
        else:
            times = _to_epoch_seconds(ld.geodataframe[anim.time_col])
        animated.append((i, ld, anim, times))
    if not animated:
        raise ValueError("draw_animated_map needs at least one animated layer (animate_layer) or a TripsLayer.")

    pieces = [(np.concatenate(t) if t else np.empty(0)) if isinstance(t, list) else t for *_, t in animated]
    pieces = [p for p in pieces if p.size]
    flat = np.concatenate(pieces) if pieces else np.zeros(1)
    t0 = float(np.nanmin(flat))
    span = max((float(np.nanmax(flat)) - t0) * 1.02, 1.0)
    logger.debug("draw_animated_map: t0=%s span=%ss", t0, span)

    # --- Rebase onto the shared clock and build each layer's animator spec ----------
    specs = []
    for i, ld, anim, times in animated:
        gdf = ld.geodataframe.copy()
        if isinstance(anim, TripsAnimation):
            ts_col = getattr(ld.layer_style, "get_timestamps", "timestamps")
            gdf[ts_col] = [(t - t0).tolist() for t in times]
            specs.append(
                {
                    "id": _layer_id(ld, i),
                    "kind": "trips",
                    "cometTrail": max(span * anim.comet_ratio, 1.0),
                    "historyTrail": span * 1.20,  # >= span -> solid back to the start
                    "showHistory": anim.show_history,
                    "historyColor": list(anim.history_color),
                    "historyOpacity": anim.history_opacity,
                    "fadeHistory": anim.fade_history,
                    "head": _head_spec(anim.head),
                }
            )
        else:
            gdf["__t"] = times - t0
            specs.append({"id": _layer_id(ld, i), "kind": "window", "window": anim.window_s, "fade": anim.fade_s})
        geo_list[i] = replace(ld, geodataframe=gdf)

    if view_state is None:
        view_state = view_state_from_layers(layers=geo_list, max_zoom=max_zoom)

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

    animation_js = (
        _ANIMATION_JS.replace("__LAYER_SPECS__", json.dumps(specs))
        .replace("__SPAN__", str(span))
        .replace("__T0__", str(t0))
        .replace("__CONTROLS__", timeline.controls.model_dump_json())
        .replace("__DURATION_S__", str(timeline.duration_s))
        .replace("__FPS_LIMIT__", str(timeline.fps_limit))
        .replace("__AUTO_ROTATE_SPEED__", str(timeline.auto_rotate_speed))
    )
    html_str = html_str.replace("</body>", animation_js + "</body>")
    return html_str


_ANIMATION_JS = """
<script>
// === Timeline animation: one shared clock drives every animated layer ===
// Each spec names a layer (by id) and an animator kind; ANIMATORS turns the base
// layer + current time into the layers actually drawn this frame.
const LAYER_SPECS     = __LAYER_SPECS__;
const maxTime         = __SPAN__;
const T0              = __T0__;            // epoch seconds at timeline 0 (for the time label)
const CONTROLS        = __CONTROLS__;   // PlaybackControls
const durationSec     = __DURATION_S__;
const timePerMs       = maxTime / (durationSec * 1000);  // timeline units per wall-clock ms
const fpsInterval     = 1000 / __FPS_LIMIT__;
const autoRotateSpeed = __AUTO_ROTATE_SPEED__;  // deg/s; 0 = off
const CPU_FILTER_LAYERS = ['HexagonLayer'];      // aggregation layers: filter rows, not on the GPU

let currentTime   = 0;
let isPlaying     = true;
let lastFrameTime = 0;
let lastTickTs    = null;  // wall clock of the last advance; null -> next frame advances 0
let baseLayers    = null;
let animatedById  = null;  // layer id -> { base, spec, animator, state }
let ScatterCtor    = null; // resolved lazily
let ScenegraphCtor = null; // loaded lazily, only when a 3D head model is configured
let DataFilterCtor = null; // loaded lazily, only when a time-window layer exists
let rotateBearing  = 0;    // accumulated bearing for smooth rotation
let lastRotateTs   = 0;    // last frame timestamp used for rotation delta
let _deckViewState = null; // mirrors interactive view state so rotation + user pan compose
let userInteracting = false; // user is mid-drag/zoom -> hold auto-rotation

const needsModel  = LAYER_SPECS.some(function (s) { return s.kind === 'trips' && s.head.kind === 'model'; });
const needsDot    = LAYER_SPECS.some(function (s) { return s.kind === 'trips' && s.head.kind !== 'none'; });
const needsFilter = LAYER_SPECS.some(function (s) { return s.kind === 'window'; });
let __headReady   = !needsModel;
let __filterReady = !needsFilter;

// Resolve a deck.gl class without hard-coding the global namespace.
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

function loadScript(src) {
  return new Promise(function (resolve, reject) {
    const s = document.createElement('script');
    s.src = src; s.onload = resolve; s.onerror = reject;
    document.head.appendChild(s);
  });
}

// deck.gl's standalone module bundles replace window.deck with only their own exports,
// so load them one at a time, hand back the module, and restore the full namespace.
let _moduleChain = Promise.resolve();
function loadDeckModule(pkg) {
  const run = async function () {
    const prev = window.deck;
    const ver = (prev && prev.VERSION) || 'latest';
    try {
      await loadScript('https://unpkg.com/' + pkg + '@' + ver + '/dist.min.js');
      return window.deck;
    } finally {
      window.deck = prev;
    }
  };
  const p = _moduleChain.then(run, run);
  _moduleChain = p.catch(function () {});
  return p;
}

async function ensureScenegraph() {
  const found = resolveLayer('ScenegraphLayer');
  if (found) return found;
  try {
    const mod = await loadDeckModule('@deck.gl/mesh-layers');
    await loadScript('https://unpkg.com/@loaders.gl/gltf@^4.0.0/dist/dist.min.js');
    const reg = (window.deck && window.deck.registerLoaders) ||
                (window.loaders && window.loaders.registerLoaders);
    const GLTFLoader = window.loaders && window.loaders.GLTFLoader;
    if (reg && GLTFLoader) { try { reg([GLTFLoader]); } catch (e) {} }
    return (mod && mod.ScenegraphLayer) || null;
  } catch (e) {
    console.warn('[draw_animated_map] could not load ScenegraphLayer/glTF loader; ' +
                 'falling back to the 2D head dot.', e);
    return null;
  }
}

async function ensureDataFilter() {
  const found = resolveLayer('DataFilterExtension');
  if (found) return found;
  try {
    const mod = await loadDeckModule('@deck.gl/extensions');
    return (mod && mod.DataFilterExtension) || null;
  } catch (e) {
    console.warn('[draw_animated_map] could not load DataFilterExtension; ' +
                 'time-window layers filter rows on the CPU (no fade).', e);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Trips animator: comet over a historic trail, plus a current-position head.
// ---------------------------------------------------------------------------

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
function headAt(feature, i, t, st, head) {
  const C = feature.geometry && feature.geometry.coordinates;
  const T = feature.timestamps;
  if (!C || !T || C.length === 0) return null;
  const last = T.length - 1;
  const k = head.smoothSamples;
  function orient(g) {
    if (g.horiz >= head.minMoveM) { st.lastHeading[i] = g.heading; return { heading: g.heading, pitch: g.pitch }; }
    return { heading: st.lastHeading[i] || 0, pitch: 0 };   // stationary: hold + level
  }
  if (t <= T[0]) {
    const o = orient(headTangent(C, 0, last, k));
    return { pos: C[0], heading: o.heading, pitch: o.pitch };
  }
  if (t >= T[last]) {
    const o = orient(headTangent(C, last - 1, last, k));
    return { pos: C[last], heading: o.heading, pitch: o.pitch };
  }
  let j = st.cursors[i] || 0;
  if (t < T[j]) j = 0;                 // scrubbed backwards
  while (j < last && T[j + 1] < t) j++;
  st.cursors[i] = j;
  const t0 = T[j], t1 = T[j + 1];
  const f = (t1 > t0) ? (t - t0) / (t1 - t0) : 0;
  const a = C[j], b = C[j + 1];
  const az = a[2] || 0, bz = b[2] || 0;
  const pos = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, az + (bz - az) * f];
  const o = orient(headTangent(C, j, last, k));
  return { pos: pos, heading: o.heading, pitch: o.pitch };
}

// A feature's track colour, read through the TripsLayer's own getColor (column accessor,
// function or constant) so the head matches whatever column the layer is coloured by.
function trackColor(base, f, i) {
  const gc = base.props.getColor;
  let c = (typeof gc === 'function') ? gc(f, { index: i, data: base.props.data, target: [] }) : gc;
  if (Array.isArray(c) && Array.isArray(c[0])) c = c[0];   // per-vertex colours -> first vertex
  return Array.isArray(c) ? Array.from(c).slice(0, 3) : [255, 0, 0];
}

function headData(base, st, head, t) {
  const feats = base.props.data || [];
  const out = [];
  for (let i = 0; i < feats.length; i++) {
    const r = headAt(feats[i], i, t, st, head);
    if (!r) continue;
    const col = head.color || trackColor(base, feats[i], i);
    out.push({ position: r.pos, color: col, heading: r.heading, pitch: r.pitch });
  }
  return out;
}

function headLayer(base, st, head, t) {
  if (head.kind === 'model' && ScenegraphCtor) {
    const m = head.model;
    return new ScenegraphCtor({
      id: base.id + '-head',
      data: headData(base, st, head, t),
      scenegraph: m.uri,
      getPosition: function (d) { return d.position; },
      getOrientation: function (d) {
        const yaw = (m.faceHeading ? -d.heading : 0) + m.yawOffset;
        const pitch = m.pitch + (m.terrainPitch ? m.terrainScale * d.pitch : 0);
        return [pitch, yaw, m.roll];   // [pitch, yaw, roll] degrees
      },
      getColor: function (d) {
        return (m.useTrackColor && d.color) ? d.color : (m.tint || [255, 255, 255]);
      },
      sizeScale: m.size,
      sizeMinPixels: m.minPx,
      sizeMaxPixels: (m.maxPx == null) ? Number.MAX_SAFE_INTEGER : m.maxPx,
      _lighting: m.lighting,
      pickable: false,
      parameters: { depthTest: true },
      updateTriggers: { getPosition: t, getOrientation: t, getColor: 'head' },
    });
  }
  if (head.kind !== 'none' && ScatterCtor) {   // dot, or fallback while/if the model can't load
    return new ScatterCtor(Object.assign({}, head.dot, {
      id: base.id + '-head',
      data: headData(base, st, head, t),
      getPosition: function (d) { return d.position; },
      getFillColor: function (d) { return d.color; },
      billboard: true,
      pickable: false,
      parameters: { depthTest: false },  // always visible over the terrain
      updateTriggers: { getPosition: t, getFillColor: 'head' },
    }));
  }
  return null;
}

// Time of one row of a time-window layer; GeoJsonLayer rows may nest it under properties.
function rowTime(d) {
  if (d.__t !== undefined) return d.__t;
  return (d.properties && d.properties.__t !== undefined) ? d.properties.__t : NaN;
}

// Each animator: init(base, spec) -> state; build(base, spec, state, t) -> { layers, overlay }.
// `layers` replace the base layer in place; `overlay` is drawn on top of everything.
const ANIMATORS = {
  trips: {
    init: function (base) {
      const n = (base.props.data || []).length;
      return { cursors: new Array(n).fill(0), lastHeading: new Array(n).fill(0) };
    },
    build: function (base, spec, st, t) {
      const layers = [];
      if (spec.showHistory) {
        // Inherit the comet's width from the TripsLayer (do NOT override getWidth).
        layers.push(base.clone({
          id: base.id + '-history',
          currentTime: t,
          trailLength: spec.historyTrail,
          fadeTrail: spec.fadeHistory,
          opacity: spec.historyOpacity,
          getColor: spec.historyColor,     // constant -> the whole track is this colour
          updateTriggers: { getColor: 'history' },
        }));
      }
      layers.push(base.clone({
        id: base.id + '-comet',
        currentTime: t,
        trailLength: spec.cometTrail,
        fadeTrail: true,
        opacity: 0.98,
      }));
      const head = headLayer(base, st, spec.head, t);
      return { layers: layers, overlay: head ? [head] : [] };
    },
  },

  window: {
    init: function (base) { return { data: base.props.data || [], extensions: null }; },
    build: function (base, spec, st, t) {
      const lo = (spec.window == null) ? -1 : t - spec.window;   // the timeline starts at 0
      const gpu = DataFilterCtor && CPU_FILTER_LAYERS.indexOf(base.constructor.layerName) < 0;
      if (gpu) {
        // One extension instance for the layer's lifetime, or deck re-initialises it. The new
        // id makes deck create a fresh layer: an extension added to an existing layer never
        // gets its filter attribute, so every row would read as time 0.
        if (!st.extensions) {
          st.extensions = (base.props.extensions || []).concat([new DataFilterCtor({ filterSize: 1 })]);
        }
        return { layers: [base.clone({
          id: base.id + '-window',
          extensions: st.extensions,
          getFilterValue: rowTime,
          filterRange: [lo, t],
          filterSoftRange: [Math.min(lo + spec.fade, t), t],
        })], overlay: [] };
      }
      const rows = st.data.filter(function (d) { const v = rowTime(d); return v >= lo && v <= t; });
      return { layers: [base.clone({ data: rows })], overlay: [] };
    },
  },
};

function buildLayers() {
  const out = [], overlay = [];
  baseLayers.forEach(function (l) {
    const a = animatedById[l.id];
    if (!a) { out.push(l); return; }
    const built = a.animator.build(a.base, a.spec, a.state, currentTime);
    out.push.apply(out, built.layers);
    overlay.push.apply(overlay, built.overlay);
  });
  return out.concat(overlay);
}

function redraw() {
  if (window.deckInstance && animatedById) window.deckInstance.setProps({ layers: buildLayers() });
  updateControls();
}

// ---------------------------------------------------------------------------
// Playback bar, configured by PlaybackControls (CONTROLS). Absent parts stay null.
// ---------------------------------------------------------------------------
let controls = null;   // { root, play, slider, label, clock, speed }
const SPEEDS = CONTROLS.speeds;
let playbackRate = 1;  // viewer's speed multiplier on top of durationSec (not used by the exporter)

function formatClock(sec) {
  const s = Math.max(0, Math.round(sec));
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}

function formatTime(t) {
  if (CONTROLS.time_format !== 'elapsed' && T0 > 1e8) {   // epoch seconds -> real date (UTC)
    const iso = new Date((T0 + t) * 1000).toISOString();
    return CONTROLS.time_format === 'date' ? iso.slice(0, 10) : iso.slice(0, 16).replace('T', ' ') + ' UTC';
  }
  const h = t / 3600;  // elapsed since the start
  return h >= 48 ? (h / 24).toFixed(1) + ' d' : h.toFixed(1) + ' h';
}

function setPlaying(on) {
  if (on && currentTime >= maxTime) currentTime = 0;   // play at the end -> start over
  if (on && !isPlaying) { isPlaying = true; lastTickTs = null; requestAnimationFrame(frame); }
  if (!on) isPlaying = false;
  updateControls();
}

function updateControls() {
  if (!controls) return;
  if (controls.play) {
    controls.play.textContent = isPlaying ? '❚❚' : '▶';
    controls.play.title = isPlaying ? 'Pause' : 'Play';
  }
  if (controls.slider && document.activeElement !== controls.slider) controls.slider.value = currentTime;
  if (controls.label) controls.label.textContent = formatTime(currentTime);
  if (controls.clock) {   // playback position / total length, at the current speed
    const total = durationSec / playbackRate;
    controls.clock.textContent = formatClock(total * currentTime / maxTime) + ' / ' + formatClock(total);
  }
  if (controls.speed) controls.speed.textContent = playbackRate + '×';
}

function buildControls() {
  const root = document.createElement('div');
  root.id = 'timeline-controls';
  // Bottom: above the corner widgets (scale bar left, legend right). Top: below the title.
  const edge = CONTROLS.position === 'top' ? 'top:56px;' : 'bottom:72px;';
  root.style.cssText = 'position:absolute;left:50%;' + edge + 'transform:translateX(-50%);z-index:10;' +
    'display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:8px;' +
    'background:rgba(20,20,20,0.72);color:#fff;font:12px/1.2 system-ui,sans-serif;' +
    'max-width:min(640px,calc(100% - 32px));box-sizing:border-box;';
  if (CONTROLS.show_scrubber) root.style.width = 'min(640px,calc(100% - 32px))';
  function button(text, title, onClick) {
    const b = document.createElement('button');
    b.textContent = text; b.title = title;
    b.style.cssText = 'background:none;border:0;color:inherit;cursor:pointer;font-size:14px;padding:2px 4px;';
    b.addEventListener('click', onClick);
    root.appendChild(b);
    return b;
  }
  function text(title) {
    const s = document.createElement('span');
    s.title = title;
    s.style.cssText = 'white-space:nowrap;font-variant-numeric:tabular-nums;';
    root.appendChild(s);
    return s;
  }
  const c = { root: root, play: null, slider: null, label: null, clock: null, speed: null };
  if (CONTROLS.show_play) c.play = button('', 'Pause', function () { setPlaying(!isPlaying); });
  if (CONTROLS.show_restart) {
    button('↺', 'Restart', function () { currentTime = 0; lastTickTs = null; redraw(); setPlaying(true); });
  }
  if (CONTROLS.show_scrubber) {
    const slider = document.createElement('input');
    slider.type = 'range'; slider.min = 0; slider.max = maxTime; slider.step = maxTime / 1000;
    slider.style.cssText = 'flex:1;min-width:60px;accent-color:#fff;';
    slider.addEventListener('input', function () {
      currentTime = Number(slider.value); lastTickTs = null; redraw();
    });
    root.appendChild(slider);
    c.slider = slider;
  }
  if (CONTROLS.show_clock) {
    c.clock = text('Playback position / total length at this speed');
    c.clock.style.opacity = '0.75';
  }
  if (CONTROLS.show_time) c.label = text('Current time in the data');
  if (CONTROLS.show_speed) {
    c.speed = button('', 'Playback speed', function () {
      playbackRate = SPEEDS[(SPEEDS.indexOf(playbackRate) + 1) % SPEEDS.length];
      updateControls();
    });
    c.speed.style.minWidth = '34px';
  }
  if (!root.children.length) return;   // every part switched off
  document.body.appendChild(root);
  controls = c;
  updateControls();
}

function frame(timestamp) {
  if (!isPlaying || !window.deckInstance) return;

  if (!timestamp) timestamp = 0;
  if (timestamp - lastFrameTime < fpsInterval) {
    requestAnimationFrame(frame);
    return;
  }
  lastFrameTime = timestamp;

  // Advance by elapsed wall time so playback lasts durationSec whatever the frame rate.
  const dt = lastTickTs === null ? 0 : timestamp - lastTickTs;
  lastTickTs = timestamp;
  if (currentTime < maxTime) {
    currentTime = Math.min(maxTime, currentTime + dt * timePerMs * playbackRate);
  } else {
    setPlaying(false);        // Stop animation at the end
  }

  redraw();

  if (autoRotateSpeed !== 0 && _deckViewState) {
    const rdt = lastRotateTs ? (timestamp - lastRotateTs) : 0;
    lastRotateTs = timestamp;
    if (!userInteracting) rotateBearing = (rotateBearing + autoRotateSpeed * rdt / 1000) % 360;
    window.deckInstance.setProps({ viewState: Object.assign({}, _deckViewState, { bearing: rotateBearing }) });
  }

  requestAnimationFrame(frame);
}

// Start
const __startWhenReady = setInterval(function () {
  if (!(window.deckInstance && window.deckInstance.props && window.deckInstance.props.layers)) return;
  clearInterval(__startWhenReady);

  baseLayers = window.deckInstance.props.layers;
  animatedById = {};
  LAYER_SPECS.forEach(function (spec) {
    const base = baseLayers.find(function (l) { return l.id === spec.id; });
    if (!base) { console.warn('[draw_animated_map] animated layer not found: ' + spec.id); return; }
    const animator = ANIMATORS[spec.kind];
    animatedById[spec.id] = { base: base, spec: spec, animator: animator, state: animator.init(base, spec) };
  });

  // Rotation makes the camera controlled (frame() sets viewState), so deck stops applying
  // user input by itself: apply every pan/zoom/rotate here immediately, resume auto-rotation
  // from the user's bearing, and hold it while they are dragging.
  if (autoRotateSpeed !== 0) {
    const ivs = window.deckInstance.props && window.deckInstance.props.initialViewState;
    if (ivs) {
      _deckViewState = Object.assign({}, ivs);
      rotateBearing  = _deckViewState.bearing || 0;
    }
    const _origOnVS = window.deckInstance.props.onViewStateChange;
    window.deckInstance.setProps({
      onViewStateChange: function (params) {
        const s = params.interactionState || {};
        userInteracting = !!(s.isDragging || s.isPanning || s.isRotating || s.isZooming);
        _deckViewState = params.viewState;
        rotateBearing  = params.viewState.bearing || 0;
        window.deckInstance.setProps({ viewState: _deckViewState });
        if (_origOnVS) _origOnVS(params);
      }
    });
  }

  if (needsDot) {
    ScatterCtor = resolveLayer('ScatterplotLayer');
    if (!ScatterCtor) {
      console.warn('[draw_animated_map] ScatterplotLayer constructor not found; ' +
                   '2D head marker disabled. Trail animation is unaffected.');
    }
  }
  if (needsModel) {
    ensureScenegraph().then(function (C) { ScenegraphCtor = C; __headReady = true; redraw(); });
  }
  if (needsFilter) {
    ensureDataFilter().then(function (C) { DataFilterCtor = C; __filterReady = true; redraw(); });
  }

  if (CONTROLS.visible) buildControls();
  requestAnimationFrame(frame);
}, 200);

// --- Deterministic render bridge (used by the server-side MP4 exporter) ---------
// Lets a headless driver pause autoplay and paint an exact frame at time t.
window.__tripsAnim = {
  get ready()    { return !!(window.deckInstance && animatedById); },
  // Every lazily loaded piece (3D head model, DataFilterExtension) is in place.
  get headReady() { return __headReady && __filterReady; },
  get span()     { return maxTime; },
  get durationSec() { return durationSec; },
  get specs()    { return LAYER_SPECS; },  // animated layers (id, kind, window), for the exporter's camera
  pause() { setPlaying(false); },
  play()  { setPlaying(true); },
  renderAt(t) {
    isPlaying = false;
    currentTime = Math.max(0, Math.min(maxTime, t));
    redraw();
  }
};
</script>
"""


@register()
def create_timeline_animation(
    duration_s: Annotated[
        float,
        AdvancedField(
            gt=0, default=30.0, description="Playback length in seconds, from the start of the timeline to its end."
        ),
    ] = 30.0,
    fps_limit: Annotated[float, AdvancedField(default=30.0, gt=0)] = 30.0,
    controls: Annotated[
        PlaybackControls,
        AdvancedField(default=PlaybackControls(), description="The playback bar (create_playback_controls)."),
    ] = PlaybackControls(),
    auto_rotate_speed: Annotated[
        float,
        AdvancedField(
            default=0.0,
            description="Camera rotation speed in degrees per second. "
            "0 = off; positive = clockwise; negative = counter-clockwise.",
        ),
    ] = 0.0,
) -> Annotated[TimelineAnimation, Field()]:
    """Construct the shared clock for draw_animated_map."""
    return TimelineAnimation(
        duration_s=duration_s,
        fps_limit=fps_limit,
        controls=controls,
        auto_rotate_speed=auto_rotate_speed,
    )


@register()
def create_playback_controls(
    visible: Annotated[bool, AdvancedField(default=True, description="Show the playback bar at all.")] = True,
    show_play: Annotated[bool, AdvancedField(default=True, description="Play/pause button.")] = True,
    show_restart: Annotated[bool, AdvancedField(default=True, description="Restart button.")] = True,
    show_scrubber: Annotated[bool, AdvancedField(default=True, description="Slider for jumping to any moment.")] = True,
    show_clock: Annotated[
        bool, AdvancedField(default=True, description="Playback position / total length, e.g. 0:12 / 0:30.")
    ] = True,
    show_time: Annotated[bool, AdvancedField(default=True, description="Current time in the data.")] = True,
    time_format: Annotated[
        Literal["datetime", "date", "elapsed"],
        AdvancedField(
            default="date",
            description="Data time as 'datetime' (2024-01-01 06:00 UTC), 'date', or 'elapsed' since the start.",
        ),
    ] = "date",
    show_speed: Annotated[bool, AdvancedField(default=True, description="Button cycling through `speeds`.")] = True,
    speeds: Annotated[
        list[Annotated[float, Field(gt=0)]],
        AdvancedField(default=[0.5, 1, 2, 4], min_length=1, description="Speed multipliers to cycle through."),
    ] = [0.5, 1, 2, 4],
    position: Annotated[
        Literal["bottom", "top"], AdvancedField(default="bottom", description="Where the bar sits on the map.")
    ] = "bottom",
) -> Annotated[PlaybackControls, Field()]:
    """Construct the playback bar config for create_timeline_animation."""
    return PlaybackControls(
        visible=visible,
        show_play=show_play,
        show_restart=show_restart,
        show_scrubber=show_scrubber,
        show_clock=show_clock,
        show_time=show_time,
        time_format=time_format,
        show_speed=show_speed,
        speeds=speeds,
        position=position,
    )


@register()
def create_trips_animation(
    head: Annotated[
        HeadMarker,
        AdvancedField(
            default=DotMarker(),
            discriminator="marker",
            title="Marker icon",
            description="Marker drawn at each subject's current position: a flat dot, a preset 3D animal, "
            "your own 3D glTF/GLB model, or none.",
        ),
    ] = DotMarker(),
    comet_ratio: Annotated[
        float, AdvancedField(default=0.3, gt=0, le=1, description="Comet-tail length as a fraction of the time span.")
    ] = 0.3,
    show_history: Annotated[bool, AdvancedField(default=True)] = True,
    history_color: Annotated[
        tuple[int, int, int], AdvancedField(default=(255, 255, 255), json_schema_extra={"items": {"type": "integer"}})
    ] = (255, 255, 255),
    history_opacity: Annotated[float, AdvancedField(default=0.85, ge=0, le=1)] = 0.85,
    fade_history: Annotated[bool, AdvancedField(default=False)] = False,
) -> Annotated[TripsAnimation, Field()]:
    """Construct a TripsAnimation for animate_layer (TripsLayer only)."""
    return TripsAnimation(
        comet_ratio=comet_ratio,
        show_history=show_history,
        history_color=history_color,
        history_opacity=history_opacity,
        fade_history=fade_history,
        head=head,
    )


@register()
def create_time_window_animation(
    time_col: Annotated[str, Field(description="Column holding each row's time (datetime or epoch seconds).")],
    window_s: Annotated[
        Annotated[float, Field(gt=0)] | SkipJsonSchema[None],
        Field(description="Seconds of data visible behind the current time. None -> everything up to now."),
    ] = None,
    fade_s: Annotated[
        float, AdvancedField(default=0.0, ge=0, description="Seconds over which rows fade out before leaving.")
    ] = 0.0,
) -> Annotated[TimeWindowAnimation, Field()]:
    """Construct a TimeWindowAnimation for animate_layer (any layer with a time column)."""
    return TimeWindowAnimation(time_col=time_col, window_s=window_s, fade_s=fade_s)


@register()
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


@register()
def set_basemap_urls(
    elevation_data: Annotated[
        PydeckString,
        AdvancedField(
            default=DEFAULT_TERRAIN_URL,
            description="URL template (or single image) for the RGB-encoded elevation tiles.",
        ),
    ] = DEFAULT_TERRAIN_URL,
    texture: Annotated[
        PydeckString | SkipJsonSchema[None],
        AdvancedField(default=SURFACE, description="URL template for tiles draped over the terrain."),
    ] = SURFACE,
) -> dict:
    """Advanced elevation/texture tile URLs for a custom basemap.

    Feed this task's return value into set_basemap_option's `basemap.tile_urls` field (only
    used when `preset: custom`) so these surface as their own advanced, collapsed fields on
    the config form instead of being bundled directly into the Custom basemap variant.
    """
    return {"elevation_data": elevation_data, "texture": texture}


@register()
def set_basemap_option(
    basemap: Annotated[
        BasemapOption,
        Field(
            description="Elevation + texture tile source (+ optional elevation decoder). Set this "
            "once and reuse its return value for both create_terrain_layer's basemap and "
            "create_terrain_sampling, so the rendered mesh and the sampled trip "
            "elevations always agree."
        ),
    ] = DefaultBasemap(),
) -> Annotated[BasemapOption, Field()]:
    """Pass through a basemap selection so multiple tasks can share one workflow step."""
    return basemap

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
from ._animate import _CAM_HELPER, camera_panel_schema
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
    utc_offset_hours: float = Field(
        default=0.0,
        ge=-14,
        le=14,
        description="Hours ahead of UTC to show the data time in, e.g. 3 for East Africa, so the clock matches "
        "the local day and the sun lighting. 0 = UTC.",
    )
    show_speed: bool = Field(default=True, description="Button cycling through `speeds`.")
    speeds: list[Annotated[float, Field(gt=0)]] = Field(
        default=[0.5, 1, 2, 4],
        min_length=1,
        description="Speed multipliers the speed button cycles through. Playback starts at 1x.",
    )
    position: Literal["bottom", "top"] = Field(default="bottom", description="Where the bar sits on the map.")
    show_camera: bool = Field(
        default=True,
        description="Camera button: opens a panel to design and preview the video camera (including keyframes) "
        "and copy its settings into render_animation.",
    )


class TimelineAnimation(BaseModel):
    """The shared clock for draw_animated_map. Knows nothing about individual layers;
    each animated layer carries its own TripsAnimation / TimeWindowAnimation."""

    duration_s: float = Field(
        default=30.0,
        gt=0,
        description="Playback length in seconds, from the start of the timeline to its end.",
    )
    fps_limit: float = Field(
        default=60.0,
        gt=0,
        description="Maximum animation frames per second.",
    )
    controls: PlaybackControls = Field(
        default_factory=PlaybackControls,
        description="The playback bar.",
    )
    sun_lighting: bool = Field(
        default=True,
        description="Light the 3D terrain by the sun's real position at the current data time and place, "
        "so hills are shaded through the day and the map darkens at night. Needs date-time data.",
    )
    night_brightness: float = Field(
        default=0.45,
        ge=0,
        le=1,
        description="How bright the terrain stays at night with sun lighting on (0 = black, 1 = no darkening).",
    )
    day_night_smoothing_s: float = Field(
        default=0.3,
        ge=0,
        le=10,
        description="Seconds of playback the sun lighting is averaged over, either side of the current moment, "
        "so day and night fade gently when many days play quickly. 0 = follow the sun exactly. When a whole "
        "day plays in under ~2.5 s, day and night fade out and the map stays in daylight.",
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
    # --- Movement ---
    smooth_turning: bool = Field(
        default=True,
        description="Turn the marker smoothly, from its direction of travel averaged over a short time, "
        "instead of snapping to a new direction at every GPS fix.",
    )
    turn_smoothing: float = Field(
        default=3.0,
        gt=0,
        le=10,
        description="How long the direction is averaged over: this many typical intervals between fixes, "
        "and at least this many tenths of a second of playback. Higher is smoother but turns later.",
    )
    curve_paths: bool = Field(
        default=False,
        description="Draw curved paths through the fixes instead of straight lines. Positions between fixes "
        "are estimated, not recorded.",
    )
    show_gaps: bool = Field(
        default=True,
        description="Show long stretches with no fixes as gaps: the trail breaks, a faint line joins the ends, "
        "and the marker is replaced by a 'no data' tag, instead of gliding across in a straight line.",
    )
    gap_hours: float = Field(default=12.0, gt=0, description="Hours without a fix that count as a gap.")
    # --- Look ---
    glow: bool = Field(default=True, description="A soft glow under the bright comet tail.")
    glow_width: float = Field(default=3.0, gt=1, le=10, description="Glow width as a multiple of the trail width.")
    glow_opacity: float = Field(default=0.25, gt=0, le=1, description="Opacity of the glow.")
    pulse: bool = Field(default=True, description="A ring that pulses out from each marker.")
    pulse_period_s: float = Field(default=1.5, gt=0, description="Seconds of playback per pulse.")
    label_column: str = Field(
        default="",
        description="Column whose value is written above each marker, e.g. subject_name. Empty for no labels.",
    )
    # --- Interaction (interactive map only) ---
    hover_info: bool = Field(
        default=True,
        description="Hovering a marker shows its label, the data time and its current speed.",
    )
    click_to_follow: bool = Field(
        default=True,
        description="Clicking a marker makes the camera follow it; drag the map, click elsewhere or press Esc to stop.",
    )


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
            label_col = anim.label_column.strip()
            if label_col and label_col not in gdf.columns:
                logger.warning("draw_animated_map: label_column %r not in the layer's data; no labels.", label_col)
                label_col = ""
            # Hover text names the track by the label column, else the legend's label column.
            tip_col = label_col or getattr(ld.legend, "label_column", None) or ""
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
                    "smoothTurning": anim.smooth_turning,
                    "turnSmoothing": anim.turn_smoothing,
                    "curvePaths": anim.curve_paths,
                    "gapS": anim.gap_hours * 3600 if anim.show_gaps else None,
                    "glow": {"width": anim.glow_width, "opacity": anim.glow_opacity} if anim.glow else None,
                    "pulsePeriod": anim.pulse_period_s if anim.pulse else None,
                    "labelCol": label_col,
                    "tipCol": tip_col if tip_col in gdf.columns else "",
                    "hoverInfo": anim.hover_info,
                    "clickToFollow": anim.click_to_follow,
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
        .replace(
            "__SUN__",
            json.dumps(
                {
                    "on": timeline.sun_lighting,
                    "night": timeline.night_brightness,
                    "smoothS": timeline.day_night_smoothing_s,
                }
            ),
        )
    )
    if timeline.controls.visible and timeline.controls.show_camera:
        animation_js += f"<script>{_CAM_HELPER}</script>" + _CAMERA_PANEL_JS.replace(
            "__CAMERA_SCHEMA__", json.dumps(camera_panel_schema()).replace("</", "<\\/")
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
const SUN             = __SUN__;            // { on, night, smoothS }: sun lighting at the data time
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
let TextCtor  = null;      // labels and 'no data' tags
let LineCtor  = null;      // faint lines across data gaps
let sunLight  = null;      // DirectionalLight that sunAt() moves with the clock (SUN.on)
let following = null;      // { layerId, fi }: the marker the camera follows (click_to_follow)

const needsModel  = LAYER_SPECS.some(function (s) { return s.kind === 'trips' && s.head.kind === 'model'; });
// ScatterplotLayer draws the dot marker and the pulse ring
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

// Index j of the segment [T[j], T[j+1]] holding t. Starts from the last answer, so normal
// playback is O(1); a jump (scrubbing, or the exporter starting mid-clip) is a binary search.
function seek(T, t, j) {
  const last = T.length - 1;
  if (last < 1) return 0;
  if (!(j >= 0 && j < last)) j = 0;
  if (T[j] <= t && t <= T[j + 1]) return j;
  if (j + 2 <= last && T[j + 1] <= t && t <= T[j + 2]) return j + 1;
  let lo = 0, hi = last - 1;                       // last j with T[j] <= t
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (T[mid] <= t) lo = mid; else hi = mid - 1; }
  return lo;
}

function median(xs) {
  if (!xs.length) return 0;
  const s = xs.slice().sort(function (a, b) { return a - b; });
  return s[s.length >> 1];
}

// Centripetal Catmull-Rom point between p1 and p2 (u in [0,1]); centripetal so uneven fix
// spacing never makes the curve loop or overshoot. Distances use cos(lat)-scaled degrees.
function crPoint(p0, p1, p2, p3, u) {
  const kx = Math.cos(p1[1] * Math.PI / 180);
  function knot(a, b) { return Math.max(Math.sqrt(Math.hypot((b[0] - a[0]) * kx, b[1] - a[1])), 1e-9); }
  const t0 = 0, t1 = t0 + knot(p0, p1), t2 = t1 + knot(p1, p2), t3 = t2 + knot(p2, p3);
  const t = t1 + (t2 - t1) * u;
  function lerp(a, b, ta, tb) {
    const w = (t - ta) / (tb - ta);
    return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, (a[2] || 0) + ((b[2] || 0) - (a[2] || 0)) * w];
  }
  const A1 = lerp(p0, p1, t0, t1), A2 = lerp(p1, p2, t1, t2), A3 = lerp(p2, p3, t2, t3);
  return lerp(lerp(A1, A2, t0, t2), lerp(A2, A3, t1, t3), t1, t2);
}

// One feature's track as the animator uses it: coordinates C and times T (curved if asked),
// which segments are gaps, and the typical interval between fixes (for turn smoothing).
function prepTrack(f, spec, subdiv) {
  const C0 = f.geometry && f.geometry.coordinates, T0s = f.timestamps;
  if (!C0 || !T0s || !C0.length) return null;
  const n = T0s.length, dts = [], lens = [];
  for (let j = 0; j + 1 < n; j++) {
    dts.push(T0s[j + 1] - T0s[j]);
    const kx = 111320 * Math.cos(C0[j][1] * Math.PI / 180);
    lens.push(Math.hypot((C0[j + 1][0] - C0[j][0]) * kx, (C0[j + 1][1] - C0[j][1]) * 110540));
  }
  const gap0 = dts.map(function (d) { return spec.gapS != null && d > spec.gapS ? 1 : 0; });
  const med = median(dts.filter(function (d, j) { return !gap0[j] && d > 0; }));
  const medLen = median(lens.filter(function (l, j) { return !gap0[j]; }));   // typical step between fixes
  if (!spec.curvePaths || subdiv < 2) return { C: C0, T: T0s, gap: gap0, med: med, medLen: medLen };
  const C = [C0[0]], T = [T0s[0]], gap = [];
  for (let j = 0; j + 1 < n; j++) {
    if (!gap0[j]) {                                    // curve within a run; never across a gap
      const p0 = (j > 0 && !gap0[j - 1]) ? C0[j - 1] : C0[j];
      const p3 = (j + 2 < n && !gap0[j + 1]) ? C0[j + 2] : C0[j + 1];
      for (let s = 1; s < subdiv; s++) {
        const u = s / subdiv;
        C.push(crPoint(p0, C0[j], C0[j + 1], p3, u));
        T.push(T0s[j] + u * dts[j]);
        gap.push(0);
      }
    }
    C.push(C0[j + 1]); T.push(T0s[j + 1]); gap.push(gap0[j]);
  }
  return { C: C, T: T, gap: gap, med: med, medLen: medLen };
}

// The trail rows drawn by the TripsLayer copies: each feature split at its gaps (so no line is
// drawn across one), with curved coordinates if asked. Plain features pass through untouched.
function trailRows(feats, tracks, spec) {
  const out = [];
  feats.forEach(function (f, i) {
    const tr = tracks[i];
    const hasGap = tr && tr.gap.some(function (g) { return g; });
    if (!tr || (!hasGap && tr.C === (f.geometry && f.geometry.coordinates))) { out.push(f); return; }
    let a = 0;
    for (let j = 0; j <= tr.gap.length; j++) {
      if (j < tr.gap.length && !tr.gap[j]) continue;
      if (j > a) {                                     // run of points a..j (>= 2 points)
        out.push(Object.assign({}, f, {
          geometry: Object.assign({}, f.geometry, { coordinates: tr.C.slice(a, j + 1) }),
          timestamps: tr.T.slice(a, j + 1),
        }));
      }
      a = j + 1;
    }
  });
  return out;
}

// Direction of travel around t from the NET movement over [t - W, t + W], each segment counted
// by the share of it inside the window. It changes continuously with t (no snap at each fix)
// and depends only on t, so the exported video matches the browser. Gaps are not crossed.
function windowHeading(tr, j, t, W) {
  const C = tr.C, T = tr.T, last = T.length - 1;
  const lo = t - W, hi = t + W;
  const kx = 111320 * Math.cos(C[j][1] * Math.PI / 180), ky = 110540;
  let ex = 0, ny = 0, dz = 0;
  function add(s) {
    const a = T[s], b = T[s + 1], span = b - a;
    const w = span > 0 ? Math.max(0, Math.min(b, hi) - Math.max(a, lo)) / span : 1;
    ex += (C[s + 1][0] - C[s][0]) * kx * w;
    ny += (C[s + 1][1] - C[s][1]) * ky * w;
    dz += ((C[s + 1][2] || 0) - (C[s][2] || 0)) * w;
  }
  add(j);
  for (let s = j - 1; s >= 0 && T[s + 1] > lo && !tr.gap[s]; s--) add(s);
  for (let s = j + 1; s < last && T[s] < hi && !tr.gap[s]; s++) add(s);
  const horiz = Math.hypot(ex, ny);
  const heading = horiz > 0 ? Math.atan2(ex, ny) * 180 / Math.PI : 0;
  const pitch = horiz > 0 ? Math.max(-20, Math.min(20, Math.atan2(dz, horiz) * 180 / Math.PI)) : 0;
  return { heading: heading, pitch: pitch, horiz: horiz };
}

const TURN_PLAYBACK_S = 0.1;   // playback seconds per unit of turn_smoothing, as a floor on the window

// A feature's marker at time t: position, orientation and speed, or { gap } while t is inside a
// data gap. When the subject is essentially stationary we HOLD the last good heading and zero
// the pitch, so the model doesn't spin or tip over while milling in place (e.g. crop-raiding).
function headAt(tr, i, t, st, head, spec) {
  if (!tr) return null;
  const C = tr.C, T = tr.T, last = T.length - 1;
  function orient(g) {
    if (g.horiz >= head.minMoveM) { st.lastHeading[i] = g.heading; return { heading: g.heading, pitch: g.pitch }; }
    return { heading: st.lastHeading[i] || 0, pitch: 0 };   // stationary: hold + level
  }
  function heading(j) {
    if (last < 1) return { heading: 0, pitch: 0, horiz: 0 };
    if (!spec.smoothTurning) return headTangent(C, j, last, head.smoothSamples);
    // While milling about, net movement over the window is GPS jitter with a random direction.
    // Widen the window until it holds a real step's worth of movement; else count as stationary.
    const enough = Math.max(head.minMoveM, 0.5 * tr.medLen);
    // At least TURN_PLAYBACK_S of playback either side too: on fast playback one frame spans
    // several fixes, and a window of a few fixes would still turn the marker on every frame.
    const playbackFloor = TURN_PLAYBACK_S * spec.turnSmoothing * maxTime / durationSec;
    let W = Math.max(tr.med * spec.turnSmoothing, playbackFloor, 1e-6), g = null;
    for (let k = 0; k < 4; k++, W *= 2) {
      g = windowHeading(tr, j, t, W);
      if (g.horiz >= enough) return g;
    }
    return { heading: g.heading, pitch: g.pitch, horiz: 0 };   // -> hold the last heading
  }
  if (t <= T[0] || last < 1) {
    const o = orient(heading(0));
    return { pos: C[0], heading: o.heading, pitch: o.pitch, speed: NaN };
  }
  if (t >= T[last]) {
    const o = orient(heading(last - 1));
    return { pos: C[last], heading: o.heading, pitch: o.pitch, speed: NaN };
  }
  const j = seek(T, t, st.cursors[i]);
  st.cursors[i] = j;
  const t0 = T[j], t1 = T[j + 1];
  if (tr.gap[j]) return { gap: true, pos: C[j], gapS: t1 - t0 };
  const f = (t1 > t0) ? (t - t0) / (t1 - t0) : 0;
  const a = C[j], b = C[j + 1];
  const az = a[2] || 0, bz = b[2] || 0;
  const pos = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, az + (bz - az) * f];
  const kx = 111320 * Math.cos(a[1] * Math.PI / 180);
  const metres = Math.hypot((b[0] - a[0]) * kx, (b[1] - a[1]) * 110540);
  const o = orient(heading(j));
  return { pos: pos, heading: o.heading, pitch: o.pitch, speed: t1 > t0 ? metres / (t1 - t0) * 3.6 : NaN };
}

// A feature's track colour, read through the TripsLayer's own getColor (column accessor,
// function or constant) so the head matches whatever column the layer is coloured by.
function trackColor(base, f, i) {
  const gc = base.props.getColor;
  let c = (typeof gc === 'function') ? gc(f, { index: i, data: base.props.data, target: [] }) : gc;
  if (Array.isArray(c) && Array.isArray(c[0])) c = c[0];   // per-vertex colours -> first vertex
  return Array.isArray(c) ? Array.from(c).slice(0, 3) : [255, 0, 0];
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
  });
}

function formatDuration(s) {
  const h = s / 3600;
  return h >= 48 ? (h / 24).toFixed(1) + ' d' : h >= 1 ? h.toFixed(1) + ' h' : Math.round(s / 60) + ' min';
}

// Markers at time t (st.heads) and the subjects currently inside a data gap (st.gapNow).
function headData(base, st, spec, t) {
  const feats = base.props.data || [];
  const head = spec.head;
  const heads = [], gapNow = [];
  for (let i = 0; i < feats.length; i++) {
    const r = headAt(st.tracks[i], i, t, st, head, spec);
    if (!r) continue;
    const label = st.labels[i];
    if (r.gap) {
      gapNow.push({ position: r.pos, text: (label ? label + ' · ' : '') + 'no data ' + formatDuration(r.gapS) });
      continue;
    }
    const col = head.color || st.colors[i];
    const d = { position: r.pos, color: col, heading: r.heading, pitch: r.pitch, fi: i, label: label };
    if (spec.hoverInfo) {
      const name = st.tips[i] || ('Track ' + (i + 1));
      d.__tip = '<b>' + escapeHtml(name) + '</b><br>' + escapeHtml(formatTime(t)) +
                (isFinite(r.speed) ? '<br>' + r.speed.toFixed(r.speed < 10 ? 1 : 0) + ' km/h' : '');
    }
    heads.push(d);
  }
  st.heads = heads;
  st.gapNow = gapNow;
  return heads;
}

// Pixel radius the pulse ring starts from: the dot's radius, or a fixed size for 3D models.
function markerRadiusPx(head) {
  return (head.kind === 'dot' && head.dot.radiusUnits === 'pixels') ? head.dot.getRadius : 14;
}

// Pulse, label and gap layers drawn over the markers. All depend on t only (the pulse phase is
// playback seconds), so a paused frame and an exported frame look the same.
function decorLayers(base, st, spec, t) {
  const out = [];
  const heads = st.heads || [];
  if (spec.pulsePeriod && ScatterCtor && spec.head.kind !== 'none' && heads.length) {
    const playbackS = t / maxTime * durationSec;
    const phase = (playbackS / spec.pulsePeriod) % 1;
    out.push(new ScatterCtor({
      id: base.id + '-pulse',
      data: heads,
      getPosition: function (d) { return d.position; },
      getRadius: markerRadiusPx(spec.head),
      radiusUnits: 'pixels',
      radiusScale: 1.1 + 1.9 * phase,
      filled: false,
      stroked: true,
      getLineColor: function (d) { return d.color; },
      getLineWidth: 2,
      lineWidthUnits: 'pixels',
      opacity: 0.85 * (1 - phase),
      billboard: true,
      pickable: false,
      parameters: { depthTest: false },
      updateTriggers: { getPosition: t, getLineColor: 'head' },
    }));
  }
  if (LineCtor && st.gapLines.length) {
    const lines = st.gapLines.filter(function (g) { return g.t0 <= t; });   // once the gap has begun
    out.push(new LineCtor({
      id: base.id + '-gaps',
      data: lines,
      getSourcePosition: function (d) { return d.from; },
      getTargetPosition: function (d) { return d.to; },
      getColor: function (d) { return d.color; },
      getWidth: 1.5,
      widthUnits: 'pixels',
      opacity: 0.45,
      pickable: false,
    }));
  }
  if (TextCtor) {
    const texts = [];
    if (spec.labelCol) {
      heads.forEach(function (d) { if (d.label) texts.push({ position: d.position, text: d.label }); });
    }
    (st.gapNow || []).forEach(function (g) { texts.push({ position: g.position, text: g.text, gap: true }); });
    if (texts.length) {
      const lift = spec.head.kind === 'model' ? 44 : markerRadiusPx(spec.head) + 12;   // clear a 3D model's height
      out.push(new TextCtor({
        id: base.id + '-labels',
        data: texts,
        getPosition: function (d) { return d.position; },
        getText: function (d) { return d.text; },
        getSize: 12,
        sizeUnits: 'pixels',
        getPixelOffset: function (d) { return d.gap ? [0, 0] : [0, -lift]; },
        getColor: function (d) { return d.gap ? [235, 235, 235, 255] : [255, 255, 255, 255]; },
        background: true,
        getBackgroundColor: function (d) { return d.gap ? [90, 90, 90, 200] : [20, 20, 20, 185]; },
        backgroundPadding: [5, 3],
        fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
        fontWeight: 600,
        characterSet: 'auto',
        billboard: true,
        pickable: false,
        parameters: { depthTest: false },
        updateTriggers: { getPosition: t, getText: t },
      }));
    }
  }
  return out;
}

function headLayer(base, st, spec, t) {
  const head = spec.head;
  const interactive = spec.hoverInfo || spec.clickToFollow;
  const onClick = spec.clickToFollow
    ? function (info) { if (info && info.object) { startFollow(base.id, info.object.fi); return true; } return false; }
    : null;
  if (head.kind === 'model' && ScenegraphCtor) {
    const m = head.model;
    return new ScenegraphCtor({
      id: base.id + '-head',
      data: headData(base, st, spec, t),
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
      pickable: interactive,
      onClick: onClick,
      parameters: { depthTest: true },
      updateTriggers: { getPosition: t, getOrientation: t, getColor: 'head' },
    });
  }
  if (head.kind !== 'none' && ScatterCtor) {   // dot, or fallback while/if the model can't load
    return new ScatterCtor(Object.assign({}, head.dot, {
      id: base.id + '-head',
      data: headData(base, st, spec, t),
      getPosition: function (d) { return d.position; },
      getFillColor: function (d) { return d.color; },
      billboard: true,
      pickable: interactive,
      onClick: onClick,
      parameters: { depthTest: false },  // always visible over the terrain
      updateTriggers: { getPosition: t, getFillColor: 'head' },
    }));
  }
  headData(base, st, spec, t);   // no marker drawn, but labels, gap tags and follow still need positions
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
    init: function (base, spec) {
      const feats = base.props.data || [];
      const n = feats.length;
      // Curved paths add (subdiv - 1) points per segment; cap the total so big layers stay fast.
      const points = feats.reduce(function (s, f) { return s + ((f.timestamps || []).length); }, 0);
      const subdiv = spec.curvePaths ? Math.max(1, Math.min(6, Math.floor(150000 / Math.max(points, 1)))) : 1;
      const tracks = feats.map(function (f) { return prepTrack(f, spec, subdiv); });
      const colors = feats.map(function (f, i) { return trackColor(base, f, i); });
      function col(name) {
        return feats.map(function (f) { const v = name ? f[name] : null; return v == null ? '' : String(v); });
      }
      // Faint lines across data gaps, in the history colour (or the track's colour without history).
      const gapLines = [];
      tracks.forEach(function (tr, i) {
        if (!tr) return;
        tr.gap.forEach(function (g, j) {
          if (g) gapLines.push({ from: tr.C[j], to: tr.C[j + 1], t0: tr.T[j],
                                 color: spec.showHistory ? spec.historyColor : colors[i] });
        });
      });
      return {
        cursors: new Array(n).fill(0), lastHeading: new Array(n).fill(0),
        tracks: tracks, colors: colors, labels: col(spec.labelCol), tips: col(spec.tipCol),
        trailData: trailRows(feats, tracks, spec), gapLines: gapLines, heads: [], gapNow: [],
      };
    },
    build: function (base, spec, st, t) {
      const layers = [];
      const data = st.trailData;   // split at gaps / curved; the same array every frame
      if (spec.showHistory) {
        // Inherit the comet's width from the TripsLayer (do NOT override getWidth).
        layers.push(base.clone({
          id: base.id + '-history',
          data: data,
          currentTime: t,
          trailLength: spec.historyTrail,
          fadeTrail: spec.fadeHistory,
          opacity: spec.historyOpacity,
          getColor: spec.historyColor,     // constant -> the whole track is this colour
          updateTriggers: { getColor: 'history' },
        }));
      }
      if (spec.glow) {   // the comet again, wider and faint, underneath it
        const p = base.props, k = spec.glow.width;
        layers.push(base.clone({
          id: base.id + '-glow',
          data: data,
          currentTime: t,
          trailLength: spec.cometTrail,
          fadeTrail: true,
          opacity: spec.glow.opacity,
          widthScale: (p.widthScale || 1) * k,
          widthMinPixels: (p.widthMinPixels || 0) * k,
          widthMaxPixels: (p.widthMaxPixels == null ? Number.MAX_SAFE_INTEGER : p.widthMaxPixels * k),
          pickable: false,
        }));
      }
      layers.push(base.clone({
        id: base.id + '-comet',
        data: data,
        currentTime: t,
        trailLength: spec.cometTrail,
        fadeTrail: true,
        opacity: 0.98,
      }));
      const head = headLayer(base, st, spec, t);
      const decor = decorLayers(base, st, spec, t);
      // pulse under the marker; labels and gap tags over it
      const pulse = decor.filter(function (l) { return /-pulse$/.test(l.id); });
      const rest = decor.filter(function (l) { return !/-pulse$/.test(l.id); });
      return { layers: layers, overlay: pulse.concat(head ? [head] : [], rest) };
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
  updateSun();   // deck reads the lights' props when it next draws
  if (window.deckInstance && animatedById) window.deckInstance.setProps({ layers: buildLayers() });
  if (window.__camPreview) window.__camPreview(currentTime);   // Camera panel's preview drives the view
  followView();
  updateControls();
}

// ---------------------------------------------------------------------------
// Sun lighting: one LightingEffect whose sun moves with the data time. Lit layers (the terrain,
// 3D markers) are shaded by the real sun position over the map; ambient light sets the night level.
// ---------------------------------------------------------------------------
// deck.gl's default material reflects 35% of ambient and 60% of diffuse light, so scale the lights
// for the terrain to come out at SUN.night of full brightness at night and ~DAY_LEVEL at midday.
// SKY is the extra ambient light from a bright sky, from civil twilight up.
const MAT_AMBIENT = 0.35, MAT_DIFFUSE = 0.6, DAY_LEVEL = 1.1, SKY = 0.12;
const DEG = Math.PI / 180;
let ambientLight = null, sunCenter = null;

// Sun altitude and azimuth (radians; azimuth from north, clockwise) at epoch ms over lat/lon in
// degrees. Low-precision solar ephemeris, good to well under a degree, which is plenty for shading.
function solarPosition(ms, lat, lon) {
  const d = (ms - 946728000000) / 86400000;                 // days since J2000.0 (2000-01-01 12:00 UTC)
  const g = (357.529 + 0.98560028 * d) * DEG;               // mean anomaly
  const q = 280.459 + 0.98564736 * d;                       // mean longitude (deg)
  const L = (q + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * DEG;   // ecliptic longitude
  const e = (23.439 - 0.00000036 * d) * DEG;                // obliquity
  const ra = Math.atan2(Math.cos(e) * Math.sin(L), Math.cos(L));
  const dec = Math.asin(Math.sin(e) * Math.sin(L));
  const gmst = (18.697374558 + 24.06570982441908 * d) * 15; // Greenwich sidereal time (deg)
  const H = (gmst + lon) * DEG - ra;                        // hour angle
  const phi = lat * DEG;
  const alt = Math.asin(Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(H));
  const az = Math.atan2(-Math.sin(H), Math.tan(dec) * Math.cos(phi) - Math.sin(phi) * Math.cos(H));
  return { alt: alt, az: az };
}

function smoothstep(a, b, x) {
  const u = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return u * u * (3 - 2 * u);
}

// The sun light at timeline time t, averaged over SUN.smoothS seconds of playback either side so
// day and night swell gently on fast playback instead of flashing. The averaged light vector gives
// flat ground exactly the average of the instantaneous lighting. Depends only on t (video = browser).
function sunAt(t) {
  const half = SUN.smoothS * maxTime / durationSec;          // data seconds either side
  const K = half > 0 ? 24 : 1;
  let vx = 0, vy = 0, vz = 0, sky = 0;
  for (let k = 0; k < K; k++) {
    const tt = K === 1 ? t : t - half + 2 * half * (k + 0.5) / K;
    const p = solarPosition((T0 + tt) * 1000, sunCenter[1], sunCenter[0]);
    const up = smoothstep(-2 * DEG, 2 * DEG, p.alt);          // soft sunrise / sunset edge
    const ca = Math.cos(p.alt);
    vx -= Math.sin(p.az) * ca * up;                            // the direction light travels
    vy -= Math.cos(p.az) * ca * up;
    vz -= Math.max(Math.sin(p.alt), 0.02) * up;
    sky += smoothstep(-6 * DEG, 4 * DEG, p.alt);               // civil twilight -> full sky light
  }
  vx /= K; vy /= K; vz /= K; sky /= K;
  const strength = Math.hypot(vx, vy, vz);
  const elev = strength > 0 ? Math.asin(-vz / strength) : 0;  // altitude of the averaged sun
  const warm = smoothstep(4 * DEG, 25 * DEG, elev);           // low sun is warm, high sun near white
  return {
    direction: strength > 0 ? [vx / strength, vy / strength, vz / strength] : [0, 0, -1],
    strength: strength,
    sky: sky,
    color: [255, Math.round(165 + 85 * warm), Math.round(105 + 135 * warm)],
  };
}

// When a whole day plays in under ~2.5 s, day and night can't be followed and averaging leaves a dull,
// half-lit map; so the effect fades out as days get shorter, down to plain daylight below 0.8 s.
const DAY_VISIBLE_S = [0.8, 2.5];
const DAYLIGHT = { direction: [-0.35, -0.35, -0.87], strength: 1, sky: 1, color: [255, 250, 240] };

function updateSun() {
  if (!sunLight) return;
  const secondsPerDay = 86400 * durationSec / maxTime;
  const v = smoothstep(DAY_VISIBLE_S[0], DAY_VISIBLE_S[1], secondsPerDay);
  const s = v > 0 ? sunAt(currentTime) : DAYLIGHT;
  function mix(a, b) { return a * v + b * (1 - v); }
  const dir = s.direction.map(function (x, i) { return mix(x, DAYLIGHT.direction[i]); });
  const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
  const night = Math.min(SUN.night, DAY_LEVEL);
  sunLight.direction = dir.map(function (x) { return x / len; });
  sunLight.color = s.color.map(function (c, i) { return Math.round(mix(c, DAYLIGHT.color[i])); });
  sunLight.intensity = mix(s.strength, 1) * Math.max(0, DAY_LEVEL - night - SKY) / MAT_DIFFUSE;
  ambientLight.intensity = (night + SKY * mix(s.sky, 1)) / MAT_AMBIENT;
}

function setupSun() {
  if (!SUN.on || !(T0 > 1e8)) return;   // needs epoch-second data times
  const Effect = resolveLayer('LightingEffect'), Ambient = resolveLayer('AmbientLight');
  const Directional = resolveLayer('DirectionalLight');
  if (!Effect || !Ambient || !Directional) {
    console.warn('[draw_animated_map] lighting classes unavailable in this deck.gl build; using default lighting.');
    return;
  }
  // The sun is placed over the middle of the data, not the camera, so a moving camera never changes it.
  let n = 0, lon = 0, lat = 0;
  Object.values(animatedById).forEach(function (a) {
    (a.state.tracks || []).forEach(function (tr) {
      if (tr && tr.C.length) { lon += tr.C[0][0]; lat += tr.C[0][1]; n++; }
    });
  });
  const v = currentView() || {};
  sunCenter = n ? [lon / n, lat / n] : [v.longitude || 0, v.latitude || 0];
  sunLight = new Directional({ color: [255, 250, 240], intensity: 1, direction: [0, 0, -1] });
  ambientLight = new Ambient({ color: [255, 255, 255], intensity: 1 });
  updateSun();
  window.deckInstance.setProps({ effects: [new Effect({ ambient: ambientLight, sun: sunLight })] });
}

// ---------------------------------------------------------------------------
// Click to follow: the camera keeps the clicked marker centred. Dragging the map, clicking
// anywhere else or Esc stops it. Interactive only: the exporter never clicks.
// ---------------------------------------------------------------------------
function currentView() {
  try {
    const v = window.deckInstance.getViewports()[0];
    return { longitude: v.longitude, latitude: v.latitude, zoom: v.zoom, pitch: v.pitch, bearing: v.bearing };
  } catch (e) { return _deckViewState; }
}

function startFollow(layerId, fi) {
  following = { layerId: layerId, fi: fi };
  followView();
}

function stopFollow() { following = null; }

function followView() {
  if (!following || window.__camPreviewOn || !window.deckInstance || !animatedById) return;
  const a = animatedById[following.layerId];
  const h = a && (a.state.heads || []).find(function (d) { return d.fi === following.fi; });
  if (!h) return;                                  // inside a data gap: hold the camera
  const v = _deckViewState || currentView();
  if (!v) return;
  _deckViewState = Object.assign({}, v, { longitude: h.position[0], latitude: h.position[1] });
  if (autoRotateSpeed === 0) window.deckInstance.setProps({ viewState: _deckViewState });
  // with auto-rotation on, frame() applies _deckViewState together with the bearing
}

// Hand the camera back to the viewer at `view`, e.g. when the Camera panel's preview ends.
function releaseCamera(view) {
  if (!window.deckInstance || !view) return;
  if (autoRotateSpeed !== 0) {
    _deckViewState = Object.assign({}, view);
    rotateBearing  = view.bearing || 0;
    window.deckInstance.setProps({ viewState: _deckViewState });
  } else {
    window.deckInstance.setProps({ viewState: null, initialViewState: Object.assign({}, view) });
  }
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
  if (CONTROLS.time_format !== 'elapsed' && T0 > 1e8) {   // epoch seconds -> real date, at the UTC offset
    const off = CONTROLS.utc_offset_hours || 0;
    const iso = new Date((T0 + t + off * 3600) * 1000).toISOString();
    if (CONTROLS.time_format === 'date') return iso.slice(0, 10);
    const m = Math.round(Math.abs(off) * 60);
    const zone = off === 0 ? 'UTC' : 'UTC' + (off < 0 ? '−' : '+') + Math.floor(m / 60) +
                 (m % 60 ? ':' + String(m % 60).padStart(2, '0') : '');
    return iso.slice(0, 16).replace('T', ' ') + ' ' + zone;
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
  if (CONTROLS.show_camera) {
    button('🎥', 'Camera: design and preview the video camera', function () {
      if (window.__camPanel) window.__camPanel.toggle();
    });
  }
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
  // Frame cap that holds an even cadence: a frame slightly early (display timing jitter) counts as
  // on time, and lateness is carried into the next interval. A strict `< fpsInterval` check skips
  // every early refresh, e.g. only ~38 fps with a 60 fps cap on a 60 Hz display (stutter).
  const sinceLast = timestamp - lastFrameTime;
  if (sinceLast < fpsInterval - Math.min(4, fpsInterval * 0.15)) {
    requestAnimationFrame(frame);
    return;
  }
  const late = sinceLast - fpsInterval;
  lastFrameTime = late > fpsInterval * 3 ? timestamp : timestamp - Math.max(0, late % fpsInterval);

  // Advance by elapsed wall time so playback lasts durationSec whatever the frame rate.
  const dt = lastTickTs === null ? 0 : timestamp - lastTickTs;
  lastTickTs = timestamp;
  if (currentTime < maxTime) {
    currentTime = Math.min(maxTime, currentTime + dt * timePerMs * playbackRate);
  } else {
    setPlaying(false);        // Stop animation at the end
  }

  redraw();

  if (autoRotateSpeed !== 0 && _deckViewState && !window.__camPreviewOn) {
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

  // Rotation and following make the camera controlled (frame()/followView() set viewState), so
  // deck stops applying user input by itself: apply every pan/zoom/rotate here immediately,
  // resume auto-rotation from the user's bearing, and hold it while they are dragging.
  // Dragging the map also stops following.
  const canFollow = LAYER_SPECS.some(function (s) { return s.kind === 'trips' && s.clickToFollow; });
  if (autoRotateSpeed !== 0 || canFollow) {
    const ivs = window.deckInstance.props && window.deckInstance.props.initialViewState;
    if (ivs) {
      _deckViewState = Object.assign({}, ivs);
      rotateBearing  = _deckViewState.bearing || 0;
    }
    const _origOnVS = window.deckInstance.props.onViewStateChange;
    window.deckInstance.setProps({
      onViewStateChange: function (params) {
        if (window.__camPreviewOn) return;   // the Camera panel's preview owns the view
        const s = params.interactionState || {};
        userInteracting = !!(s.isDragging || s.isPanning || s.isRotating || s.isZooming);
        if (following && (s.isDragging || s.isPanning)) stopFollow();
        _deckViewState = params.viewState;
        rotateBearing  = params.viewState.bearing || 0;
        window.deckInstance.setProps({ viewState: _deckViewState });
        if (following) followView();          // zooming while following stays centred
        if (_origOnVS) _origOnVS(params);
      }
    });
  }
  if (canFollow) {
    // A click that no marker handled (empty map, other layers) stops following; so does Esc.
    const _origOnClick = window.deckInstance.props.onClick;
    window.deckInstance.setProps({
      onClick: function (info, event) {
        if (following) stopFollow();
        if (_origOnClick) return _origOnClick(info, event);
      }
    });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') stopFollow(); });
  }
  if (LAYER_SPECS.some(function (s) { return s.kind === 'trips' && s.hoverInfo; })) {
    // Markers carry their own tooltip HTML (__tip); anything else keeps the map's tooltip.
    const _origTip = window.deckInstance.props.getTooltip;
    window.deckInstance.setProps({
      getTooltip: function (info) {
        const o = info && info.object;
        if (o && o.__tip) {
          return { html: o.__tip, style: { background: 'rgba(20,20,20,0.85)', color: '#fff', fontSize: '12px',
                                           padding: '6px 8px', borderRadius: '6px', lineHeight: '1.35' } };
        }
        return _origTip ? _origTip(info) : null;
      }
    });
  }

  TextCtor = resolveLayer('TextLayer');
  LineCtor = resolveLayer('LineLayer');
  setupSun();

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
  get time()     { return currentTime; },
  pause() { setPlaying(false); },
  setView(view) { releaseCamera(view); },
  play()  { setPlaying(true); },
  renderAt(t) {
    isPlaying = false;
    currentTime = Math.max(0, Math.min(maxTime, t));
    redraw();
  }
};
</script>
"""


_CAMERA_PANEL_JS = """
<script>
// === Camera panel: design the video camera here, then hand its settings to render_animation ===
// The form is built from render_animation's camera models (CAMERAS), and the preview runs the
// exporter's own camera code (window.__cam), so what you see is what the video gets -- except
// that the video refits the framing to its own size.
(function () {
  const CAMERAS = __CAMERA_SCHEMA__;
  const PREVIEW_FPS = 30;
  let views = null, lastView = null, panel = null, body = null, preview = false;

  // A camera being edited: its type, every type's field values (so switching type and back keeps
  // them), keyframes (keyframes camera), the subject to follow without keyframes, and segments
  // (sequence camera; each segment holds a camera of its own).
  function newCamera(type) {
    const cs = { type: type, values: {}, keyframes: [], subject: '', segments: [] };
    CAMERAS.forEach(function (c) {
      cs.values[c.type] = {};
      c.fields.forEach(function (f) {
        cs.values[c.type][f.name] = Array.isArray(f.default) ? f.default.slice() : f.default;
      });
    });
    return cs;
  }
  function newSegment(until, type) { return { until: until, cam: newCamera(type), open: false }; }
  const root = newCamera(CAMERAS[0].type);
  root.segments = [newSegment(0.2, 'fly_around'), newSegment(0.85, 'follow'), newSegment(1, 'fit')];

  function camera(type) { return CAMERAS.find(function (c) { return c.type === type; }); }
  function camSpan() { return window.__cam.span() || window.__tripsAnim.span; }
  function progress() { return Math.max(0, Math.min(1, window.__tripsAnim.time / camSpan())); }
  function round(x, n) { const p = Math.pow(10, n); return Math.round(x * p) / p; }
  function currentView() {
    try {
      const v = window.deckInstance.getViewports()[0];
      return { longitude: v.longitude, latitude: v.latitude, zoom: v.zoom, pitch: v.pitch, bearing: v.bearing };
    } catch (e) { return lastView; }
  }
  // Clip range [start, end] of each sequence segment.
  function segmentRanges(cs) {
    let prev = 0;
    return cs.segments.map(function (s, i) {
      const end = i === cs.segments.length - 1 ? 1 : Math.max(prev, Math.min(1, s.until));
      const r = [prev, end];
      prev = end;
      return r;
    });
  }

  // --- Settings: exactly the camera object render_animation takes -------------------------
  function settings(cs) {
    const out = { type_: cs.type }, values = cs.values[cs.type];
    // Unset values are the defaults; leaving them out keeps optional limits from rejecting null.
    Object.keys(values).forEach(function (k) { if (values[k] != null) out[k] = values[k]; });
    if (cs.type === 'keyframes') {
      if (cs.keyframes.length) {
        out.keyframes = cs.keyframes.map(function (k) {
          return { t: round(k.t, 4), lon: round(k.lon, 6), lat: round(k.lat, 6),
                   zoom: round(k.zoom, 2), pitch: round(k.pitch, 1), bearing: round(k.bearing, 1) };
        });
      } else {
        out.source = { type_: 'subject', subject: cs.subject || null };
      }
    }
    if (cs.type === 'sequence') {
      out.segments = cs.segments.map(function (s, i) {
        return { until: i === cs.segments.length - 1 ? 1 : round(s.until, 4), camera: settings(s.cam) };
      });
    }
    return out;
  }

  // Same shape as render_animation's _camera_opts; keyframes resolved as _resolve_keyframes
  // does for complete ones (sorted, bearings unwrapped for shortest turns).
  function pathOptions(cs, dims) {
    const s = settings(cs);
    const o = Object.assign({}, s, dims, { preset: s.type_, auto_keyframe_count: 12, keyframes: null });
    if (cs.type === 'keyframes') o.subject = cs.subject || null;
    delete o.type_; delete o.source;
    if (cs.keyframes.length >= 2 && cs.type === 'keyframes') {
      const K = cs.keyframes.slice().sort(function (a, b) { return a.t - b.t; }).map(function (k) {
        return Object.assign({}, k);
      });
      for (let i = 1; i < K.length; i++) {
        const d = ((K[i].bearing - K[i - 1].bearing + 180) % 360 + 360) % 360 - 180;
        K[i].bearing = K[i - 1].bearing + d;
      }
      o.keyframes = K;
    }
    if (cs.type === 'sequence') {
      o.segments = cs.segments.map(function (seg, i) {
        return { until: i === cs.segments.length - 1 ? 1 : seg.until, camera: pathOptions(seg.cam, dims) };
      });
    }
    return o;
  }

  // --- Preview: one view per frame over the whole timeline, looked up by the current time -----
  function invalidate() { views = null; if (preview) window.__camPreview(window.__tripsAnim.time); }
  window.__camPreview = function (t) {
    if (!preview) return;
    if (!views) {
      const span = camSpan(), n = Math.max(2, Math.round(window.__tripsAnim.durationSec * PREVIEW_FPS));
      const canvas = document.querySelector('#deck-container canvas');
      const dims = { fps: PREVIEW_FPS, width: canvas ? canvas.clientWidth : 1280,
                     height: canvas ? canvas.clientHeight : 720 };
      const times = [];
      for (let i = 0; i < n; i++) times.push(span * i / (n - 1));
      views = window.__cam.path(times, pathOptions(root, dims));
    }
    const v = views[Math.round(Math.max(0, Math.min(1, t / camSpan())) * (views.length - 1))];
    if (!v) return;
    lastView = v;
    window.deckInstance.setProps({ viewState: Object.assign({}, v) });
  };
  function setPreview(on) {
    preview = on;
    window.__camPreviewOn = on;
    if (on) { invalidate(); return; }
    window.__tripsAnim.setView(lastView || currentView());
  }

  // --- Panel UI ----------------------------------------------------------------------------
  const CSS = 'position:absolute;top:56px;right:12px;z-index:11;width:300px;max-height:calc(100% - 170px);' +
    'overflow:auto;padding:10px 12px;border-radius:8px;box-sizing:border-box;' +
    'background:rgba(20,20,20,0.85);color:#fff;font:12px/1.35 system-ui,sans-serif;';
  const INPUT = 'width:100%;box-sizing:border-box;margin-top:2px;padding:3px 5px;border-radius:4px;' +
    'border:1px solid rgba(255,255,255,0.25);background:rgba(255,255,255,0.08);color:#fff;font:inherit;';
  const BUTTON = 'padding:4px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.35);' +
    'background:rgba(255,255,255,0.1);color:#fff;font:inherit;cursor:pointer;';
  const SECTION = 'margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.2);';
  const NESTED = 'margin-top:6px;padding:4px 0 6px 10px;border-left:2px solid rgba(255,255,255,0.25);';

  function el(tag, css, text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text != null) e.textContent = text;
    return e;
  }
  function row(parent, label, help, input) {
    const r = el('label', 'display:block;margin-top:8px;');
    r.appendChild(el('span', 'opacity:0.8;', label));
    if (help) r.title = help;
    r.appendChild(input);
    parent.appendChild(r);
    return r;
  }
  function button(parent, text, title, onClick) {
    const b = el('button', BUTTON, text);
    b.type = 'button'; b.title = title || '';
    b.addEventListener('click', onClick);
    parent.appendChild(b);
    return b;
  }
  function select(options, value, onChange) {   // options: [[value, label], ...]
    const s = el('select', INPUT);
    options.forEach(function (o) { const opt = el('option', '', o[1]); opt.value = o[0]; s.appendChild(opt); });
    s.value = value == null ? '' : value;
    s.addEventListener('change', function () { onChange(s.value); });
    return s;
  }

  function fieldInput(f, values) {
    let input;
    if (f.type === 'boolean') {
      input = el('input', 'margin-left:6px;vertical-align:middle;');
      input.type = 'checkbox'; input.checked = !!values[f.name];
      input.addEventListener('change', function () { values[f.name] = input.checked; invalidate(); });
      return input;
    }
    if (f.type === 'enum') {
      return select(f.options.map(function (o) { return [o, o]; }), values[f.name],
                    function (v) { values[f.name] = v; invalidate(); });
    }
    if (f.name === 'layer') {   // pick one of the page's time-window layers
      const ids = window.__tripsAnim.specs.filter(function (s) { return s.kind === 'window'; })
                                          .map(function (s) { return [s.id, s.id]; });
      return select([['', 'All time-window layers']].concat(ids), values[f.name],
                    function (v) { values[f.name] = v || null; invalidate(); });
    }
    input = el('input', INPUT);
    if (f.type === 'array') {   // list of names, typed comma-separated
      input.type = 'text'; input.placeholder = 'all, or names separated by commas';
      input.value = (values[f.name] || []).join(', ');
      input.addEventListener('change', function () {
        const list = input.value.split(',').map(function (x) { return x.trim(); }).filter(Boolean);
        values[f.name] = list.length ? list : null;
        invalidate();
      });
      return input;
    }
    const numeric = f.type === 'number' || f.type === 'integer';
    input.type = numeric ? 'number' : 'text';
    if (numeric) {
      input.step = f.type === 'integer' ? '1' : 'any';
      if (f.min != null) input.min = f.min;
      if (f.max != null) input.max = f.max;
    }
    if (f.default == null) input.placeholder = 'auto';
    input.value = values[f.name] == null ? '' : values[f.name];
    input.addEventListener('change', function () {
      let v = input.value.trim();
      if (v === '') v = f.default;                     // blank -> the field's default (auto when None)
      else if (numeric) {
        v = Number(v);
        if (!isFinite(v)) v = f.default;
        else {
          if (f.min != null) v = Math.max(f.min, v);
          if (f.max != null) v = Math.min(f.max, v);
          if (f.type === 'integer') v = Math.round(v);
        }
      }
      values[f.name] = v;
      input.value = v == null ? '' : v;
      invalidate();
    });
    return input;
  }

  // Keyframes of `cs`, which runs over clip range `range`: times are stored relative to it.
  function keyframeSection(parent, cs, range) {
    const box = el('div', SECTION);
    const len = Math.max(1e-6, range[1] - range[0]);
    box.appendChild(el('div', 'font-weight:600;', 'Keyframes'));
    box.appendChild(el('div', 'opacity:0.7;margin-top:2px;',
      'Scrub to a moment, frame the map, then add it. With fewer than 2 keyframes the camera follows a subject.'));
    const subject = el('input', INPUT);
    subject.type = 'text'; subject.placeholder = 'longest-running track'; subject.value = cs.subject;
    subject.addEventListener('change', function () { cs.subject = subject.value.trim(); invalidate(); });
    row(box, 'Subject to follow', 'Value identifying the subject in any column (e.g. its name).', subject);
    const add = el('div', 'margin-top:8px;');
    button(add, '+ Add current view', 'Add a keyframe at the current time with the current view', function () {
      const v = currentView();
      if (!v) return;
      const t = Math.max(0, Math.min(1, (progress() - range[0]) / len));
      cs.keyframes.push({ t: t, lon: v.longitude, lat: v.latitude, zoom: v.zoom, pitch: v.pitch || 0,
                          bearing: v.bearing || 0 });
      cs.keyframes.sort(function (a, b) { return a.t - b.t; });
      invalidate(); render();
    });
    box.appendChild(add);
    cs.keyframes.forEach(function (k, i) {
      const r = el('div', 'display:flex;align-items:center;gap:6px;margin-top:6px;font-variant-numeric:tabular-nums;');
      r.appendChild(el('span', 'flex:1;', ((range[0] + k.t * len) * 100).toFixed(0) + '%  z' + k.zoom.toFixed(1) +
        '  tilt ' + k.pitch.toFixed(0) + '°  hdg ' + k.bearing.toFixed(0) + '°'));
      button(r, 'Go', 'Jump to this keyframe', function () {
        window.__tripsAnim.renderAt((range[0] + k.t * len) * camSpan());
        if (!preview) window.__tripsAnim.setView({ longitude: k.lon, latitude: k.lat, zoom: k.zoom,
                                                  pitch: k.pitch, bearing: k.bearing });
      });
      button(r, 'Set', 'Replace with the current view (keeps its time)', function () {
        const v = currentView();
        if (!v) return;
        Object.assign(k, { lon: v.longitude, lat: v.latitude, zoom: v.zoom, pitch: v.pitch || 0,
                           bearing: v.bearing || 0 });
        invalidate(); render();
      });
      button(r, '✕', 'Remove', function () { cs.keyframes.splice(i, 1); invalidate(); render(); });
      box.appendChild(r);
    });
    parent.appendChild(box);
  }

  function sequenceSection(parent, cs) {
    const box = el('div', SECTION);
    box.appendChild(el('div', 'font-weight:600;', 'Segments'));
    box.appendChild(el('div', 'opacity:0.7;margin-top:2px;',
      'Each segment uses its own camera until its end, blending into the next.'));
    const ranges = segmentRanges(cs);
    cs.segments.forEach(function (seg, i) {
      const last = i === cs.segments.length - 1;
      const r = el('div', 'display:flex;align-items:center;gap:6px;margin-top:8px;');
      r.appendChild(el('span', 'min-width:16px;opacity:0.8;', (i + 1) + '.'));
      const type = select(CAMERAS.filter(function (c) { return c.type !== 'sequence'; })
                                 .map(function (c) { return [c.type, c.title]; }),
                          seg.cam.type, function (v) { seg.cam.type = v; invalidate(); render(); });
      type.style.flex = '1';
      r.appendChild(type);
      if (last) {
        r.appendChild(el('span', 'opacity:0.7;white-space:nowrap;', 'to end'));
      } else {
        const until = el('input', INPUT + 'width:58px;');
        until.type = 'number'; until.min = 1; until.max = 99; until.step = 1;
        until.value = Math.round(seg.until * 100);
        until.title = 'Clip position (%) where this segment ends';
        until.addEventListener('change', function () {
          const lo = i > 0 ? cs.segments[i - 1].until : 0, hi = cs.segments[i + 1].until;
          seg.until = Math.max(lo, Math.min(hi, (Number(until.value) || 0) / 100));
          invalidate(); render();
        });
        r.appendChild(until);
        r.appendChild(el('span', 'opacity:0.7;', '%'));
      }
      button(r, seg.open ? '▾' : '▸', 'Edit this camera', function () { seg.open = !seg.open; render(); });
      if (cs.segments.length > 1) {
        button(r, '✕', 'Remove segment', function () {
          cs.segments.splice(i, 1);
          if (cs.segments.length) cs.segments[cs.segments.length - 1].until = 1;
          invalidate(); render();
        });
      }
      box.appendChild(r);
      if (seg.open) {
        const nested = el('div', NESTED);
        cameraForm(nested, seg.cam, ranges[i], true);
        box.appendChild(nested);
      }
    });
    const add = el('div', 'margin-top:8px;');
    button(add, '+ Add segment', 'Split the last segment in two', function () {
      const r = ranges[ranges.length - 1];
      cs.segments.splice(cs.segments.length - 1, 0, newSegment(round((r[0] + r[1]) / 2, 3), 'follow'));
      invalidate(); render();
    });
    box.appendChild(add);
    parent.appendChild(box);
  }

  function cameraForm(parent, cs, range, nested) {
    if (!nested) {
      row(parent, 'Camera', '', select(CAMERAS.map(function (c) { return [c.type, c.title]; }), cs.type,
                                       function (v) { cs.type = v; invalidate(); render(); }));
    }
    const cam = camera(cs.type);
    parent.appendChild(el('div', 'opacity:0.7;margin-top:4px;', cam.description));
    cam.fields.forEach(function (f) { row(parent, f.title, f.description, fieldInput(f, cs.values[cs.type])); });
    if (cs.type === 'keyframes') keyframeSection(parent, cs, range);
    if (cs.type === 'sequence') sequenceSection(parent, cs);
  }

  function copyText(text, done) {
    function fallback() {   // sandboxed frames can block the Clipboard API
      const ta = el('textarea', 'position:fixed;left:-9999px;');
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { if (document.execCommand('copy')) done(); } finally { ta.remove(); }
    }
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done, fallback);
    else fallback();
  }

  function render() {
    const scroll = panel.scrollTop;
    body.innerHTML = '';
    cameraForm(body, root, [0, 1], false);

    const actions = el('div', SECTION);
    const prev = el('input', 'margin-right:6px;vertical-align:middle;');
    prev.type = 'checkbox'; prev.checked = preview;
    prev.addEventListener('change', function () { setPreview(prev.checked); });
    const pl = el('label', 'display:block;cursor:pointer;');
    pl.appendChild(prev); pl.appendChild(document.createTextNode('Preview camera while playing'));
    actions.appendChild(pl);
    const btns = el('div', 'display:flex;gap:6px;align-items:center;margin-top:8px;');
    const note = el('span', 'opacity:0.75;');
    button(btns, 'Copy settings', "Copy, then paste into render_animation's Camera settings", function () {
      copyText(JSON.stringify(settings(root)), function () { note.textContent = 'Copied'; });
    });
    button(btns, 'Download', "Save as a file to upload as render_animation's Camera settings", function () {
      const a = el('a');
      a.href = URL.createObjectURL(new Blob([JSON.stringify(settings(root), null, 2)], { type: 'application/json' }));
      a.download = 'camera-settings.json';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    });
    btns.appendChild(note);
    actions.appendChild(btns);
    actions.appendChild(el('div', 'opacity:0.7;margin-top:6px;',
      "Use these settings in the workflow's render step (Camera settings). The video refits the framing " +
      'to its own size, so it can differ slightly from this window.'));
    body.appendChild(actions);
    panel.scrollTop = scroll;
  }

  function build() {
    panel = el('div', CSS);
    panel.id = 'camera-panel';
    panel.hidden = true;
    const head = el('div', 'display:flex;align-items:center;font-weight:600;font-size:13px;');
    head.appendChild(el('span', 'flex:1;', 'Video camera'));
    const close = button(head, '✕', 'Close', function () { panel.hidden = true; });
    close.style.cssText = 'background:none;border:0;color:#fff;cursor:pointer;font-size:14px;';
    panel.appendChild(head);
    body = el('div');
    panel.appendChild(body);
    document.body.appendChild(panel);
    render();
  }

  window.__camPanel = {
    toggle: function () { if (!panel) build(); panel.hidden = !panel.hidden; },
    get settings() { return settings(root); },
  };
})();
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
    fps_limit: Annotated[float, AdvancedField(default=60.0, gt=0)] = 60.0,
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
    sun_lighting: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Light the 3D terrain by the sun's real position at the current data time and place, "
            "so hills are shaded through the day and the map darkens at night. Needs date-time data.",
        ),
    ] = True,
    night_brightness: Annotated[
        float,
        AdvancedField(
            default=0.45,
            ge=0,
            le=1,
            description="How bright the terrain stays at night with sun lighting on (0 = black, 1 = no darkening).",
        ),
    ] = 0.45,
    day_night_smoothing_s: Annotated[
        float,
        AdvancedField(
            default=0.3,
            ge=0,
            le=10,
            description=TimelineAnimation.model_fields["day_night_smoothing_s"].description,
        ),
    ] = 0.3,
) -> Annotated[TimelineAnimation, Field()]:
    """Construct the shared clock for draw_animated_map."""
    return TimelineAnimation(
        duration_s=duration_s,
        fps_limit=fps_limit,
        controls=controls,
        auto_rotate_speed=auto_rotate_speed,
        sun_lighting=sun_lighting,
        night_brightness=night_brightness,
        day_night_smoothing_s=day_night_smoothing_s,
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
    utc_offset_hours: Annotated[
        float,
        AdvancedField(
            default=0.0, ge=-14, le=14, description=PlaybackControls.model_fields["utc_offset_hours"].description
        ),
    ] = 0.0,
    show_speed: Annotated[bool, AdvancedField(default=True, description="Button cycling through `speeds`.")] = True,
    speeds: Annotated[
        list[Annotated[float, Field(gt=0)]],
        AdvancedField(default=[0.5, 1, 2, 4], min_length=1, description="Speed multipliers to cycle through."),
    ] = [0.5, 1, 2, 4],
    position: Annotated[
        Literal["bottom", "top"], AdvancedField(default="bottom", description="Where the bar sits on the map.")
    ] = "bottom",
    show_camera: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Camera button: design and preview the video camera, then copy its settings into "
            "render_animation.",
        ),
    ] = True,
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
        utc_offset_hours=utc_offset_hours,
        show_speed=show_speed,
        speeds=speeds,
        position=position,
        show_camera=show_camera,
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
    smooth_turning: Annotated[
        bool, AdvancedField(default=True, description=TripsAnimation.model_fields["smooth_turning"].description)
    ] = True,
    turn_smoothing: Annotated[
        float,
        AdvancedField(default=3.0, gt=0, le=10, description=TripsAnimation.model_fields["turn_smoothing"].description),
    ] = 3.0,
    curve_paths: Annotated[
        bool, AdvancedField(default=False, description=TripsAnimation.model_fields["curve_paths"].description)
    ] = False,
    show_gaps: Annotated[
        bool, AdvancedField(default=True, description=TripsAnimation.model_fields["show_gaps"].description)
    ] = True,
    gap_hours: Annotated[
        float, AdvancedField(default=12.0, gt=0, description=TripsAnimation.model_fields["gap_hours"].description)
    ] = 12.0,
    glow: Annotated[
        bool, AdvancedField(default=True, description=TripsAnimation.model_fields["glow"].description)
    ] = True,
    glow_width: Annotated[
        float,
        AdvancedField(default=3.0, gt=1, le=10, description=TripsAnimation.model_fields["glow_width"].description),
    ] = 3.0,
    glow_opacity: Annotated[
        float,
        AdvancedField(default=0.25, gt=0, le=1, description=TripsAnimation.model_fields["glow_opacity"].description),
    ] = 0.25,
    pulse: Annotated[
        bool, AdvancedField(default=True, description=TripsAnimation.model_fields["pulse"].description)
    ] = True,
    pulse_period_s: Annotated[
        float, AdvancedField(default=1.5, gt=0, description=TripsAnimation.model_fields["pulse_period_s"].description)
    ] = 1.5,
    label_column: Annotated[
        str, AdvancedField(default="", description=TripsAnimation.model_fields["label_column"].description)
    ] = "",
    hover_info: Annotated[
        bool, AdvancedField(default=True, description=TripsAnimation.model_fields["hover_info"].description)
    ] = True,
    click_to_follow: Annotated[
        bool, AdvancedField(default=True, description=TripsAnimation.model_fields["click_to_follow"].description)
    ] = True,
) -> Annotated[TripsAnimation, Field()]:
    """Construct a TripsAnimation for animate_layer (TripsLayer only)."""
    return TripsAnimation(
        comet_ratio=comet_ratio,
        show_history=show_history,
        history_color=history_color,
        history_opacity=history_opacity,
        fade_history=fade_history,
        head=head,
        smooth_turning=smooth_turning,
        turn_smoothing=turn_smoothing,
        curve_paths=curve_paths,
        show_gaps=show_gaps,
        gap_hours=gap_hours,
        glow=glow,
        glow_width=glow_width,
        glow_opacity=glow_opacity,
        pulse=pulse,
        pulse_period_s=pulse_period_s,
        label_column=label_column,
        hover_info=hover_info,
        click_to_follow=click_to_follow,
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

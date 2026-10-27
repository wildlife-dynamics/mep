import os
import io
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
from ._animate import camera_panel_schema
from ecoscope.platform.tasks.results._pydeck import (
    PydeckAnnotation,
    PydeckString,
    LayerStyleBase,
    ColorAccessor,
    FloatAccessor,
    UnitType,
    LegendDefinition,
    PydeckLayerDefinition,
    LegendStyle,
    ViewState,
    LegendFromDataframe,
    PYDECK_CUSTOM_LIBRARIES,
    _model_dump_for_pydeck,
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

# TimelineAnimationWidget's bundle, built+attached to the git tag by .github/workflows/release.yml
# in https://github.com/ttemu34/my-deck-widgets (bump the tag after each release). Set
# MY_DECK_WIDGETS_URI to try a local build, e.g. http://localhost:8080/dist/bundle.js (npm run dev).
MY_DECK_WIDGETS = {
    "libraryName": "myDeckWidgets",
    "resourceUri": os.environ.get(
        "MY_DECK_WIDGETS_URI", "https://cdn.jsdelivr.net/gh/ttemu34/my-deck-widgets@v0.17.0/dist/bundle.js"
    ),
}

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
    """Flat circle at each subject's current position (deck.gl ScatterplotLayer), filled
    with that subject's track colour.

    Field names mirror PointLayerStyle. Colours are constants rather than column accessors
    because the marker's data is rebuilt in the browser every frame.
    """

    model_config = ConfigDict(json_schema_extra={"title": "Dot"})
    marker: Annotated[Literal["dot"], Field(default="dot", title="Marker icon")] = "dot"
    get_radius: Annotated[float, AdvancedField(default=2.0, gt=0)] = 2.0
    radius_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    stroked: Annotated[bool, AdvancedField(default=True)] = True
    get_line_color: Annotated[
        tuple[int, int, int], 
        AdvancedField(default=(255, 255, 255), json_schema_extra={"items": {"type": "integer"}})
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
    walk_animation: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            title="Walk animation",
            description="Name or index of the model's animation to play while the subject moves "
            "(with the trips animation's speed response on). None -> no animation.",
        ),
    ] = None
    run_animation: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            title="Run animation",
            description="Animation played while the subject moves fast (over 2.5x its typical speed). "
            "None -> the walk animation.",
        ),
    ] = None
    idle_animation: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            title="Idle animation",
            description="Animation played while the subject rests. None -> the model holds still.",
        ),
    ] = None
    animation_speed: Annotated[
        float,
        AdvancedField(
            default=1.0, gt=0, le=10, description="Playback speed of the model's animations (1 = as authored)."
        ),
    ] = 1.0

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
# (walk, run, idle) animation indices in each GLB; the giraffe, cheetah and leopard have none.
_ANIMAL_ANIMATIONS = {
    "elephant": ("12", "11", "7"),  # walk, Run, Idle1
    "lion": ("0", None, None),  # its one loop
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
        walk, run, idle = _ANIMAL_ANIMATIONS.get(self.animal, (None, None, None))
        return ScenegraphLayerDefinition(
            glb=_ANIMAL_MODEL_URL.format(self.animal),
            size_scale=scale * self.size,
            size_min_pixels=min_px * self.size,
            size_max_pixels=max_px * self.size,
            yaw_offset=yaw,
            use_track_color=self.use_track_color,
            pbr_lighting=self.pbr_lighting,
            walk_animation=walk,
            run_animation=run,
            idle_animation=idle,
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
    adaptive_speed: bool = Field(
        default=False,
        description="Pace the animation by activity: stretches where every subject rests (or has no data) "
        "play faster, so the time goes to movement. The clock still shows the real data time.",
    )
    max_speedup: float = Field(
        default=6.0,
        gt=1,
        le=50,
        description="With adaptive speed, how many times faster quiet stretches play than active ones.",
    )
    rest_speed_kmh: float = Field(
        default=0.5,
        ge=0,
        description="Below this speed (km/h) a subject counts as resting: for adaptive speed, and for the "
        "trips animation's speed response (markers shrink and stop pulsing, 3D models idle).",
    )
    fast_forward_night: bool = Field(
        default=False,
        description="With adaptive speed, also play nights faster, even if subjects move. Needs date-time data.",
    )
    ease_s: float = Field(
        default=0.6,
        ge=0,
        le=5,
        description="Seconds to ease the playback speed in and out on play, pause, speed changes and at the "
        "end of the clip, instead of jumping. Interactive map only; the video plays at a steady rate. 0 = off.",
    )
    follow_smoothing_s: float = Field(
        default=0.4,
        ge=0,
        le=5,
        description="How softly the camera follows a clicked marker: about this many seconds to catch up. "
        "0 = locked on.",
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
    label_column: str = Field(
        default="",
        description="Column whose value is written above each marker, e.g. subject_name. Empty for no labels.",
    )
    # --- Interaction (interactive map only) ---
    hover_info: bool = Field(
        default=True,
        description="Hovering a marker shows its label, the data time and its current speed.",
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
class AnimatedLayerDefinition(PydeckLayerDefinition):
    """A PydeckLayerDefinition plus the animation draw_animated_map drives it with."""

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

    altitude_mode: Literal["clamp_to_ground", "relative_to_ground", "absolute"] = Field(
        default="relative_to_ground",
        description="Height of the tracks, as Google Maps 3D altitude modes: on the ground (clamp_to_ground), "
        "`offset` metres above it (relative_to_ground), or each fix's own recorded altitude above sea level "
        "(absolute; `offset` metres above sea level where a fix has none).",
    )
    offset: float = Field(
        default=30.0,
        description="Metres above the ground (relative_to_ground), or above sea level for fixes without an "
        "altitude (absolute). Ignored by clamp_to_ground.",
    )
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
    altitude_mode: Annotated[
        Literal["clamp_to_ground", "relative_to_ground", "absolute"],
        AdvancedField(
            default="relative_to_ground",
            description="Height of the tracks: on the ground (clamp_to_ground; lines can dip into the terrain "
            "between fixes), `offset` metres above it (relative_to_ground), or each fix's recorded altitude above "
            "sea level (absolute).",
        ),
    ] = "relative_to_ground",
    offset: Annotated[
        float,
        AdvancedField(
            default=30,
            description="Metres above the ground (relative_to_ground), or above sea level for fixes without an "
            "altitude (absolute). Ignored by clamp_to_ground.",
        ),
    ] = 30.0,
    ground_elevation: Annotated[
        float, AdvancedField(default=1000.0, description="Constant ground used only if DEM sampling fails.")
    ] = 1000.0,
) -> Annotated[TerrainSampling, Field()]:
    """Creates the elevation sampling config consumed by drape_trips_on_terrain."""
    return TerrainSampling(
        elevation_data=elevation_data,
        elevation_decoder=elevation_decoder or dict(TERRARIUM_ELEVATION_DECODER),
        altitude_mode=altitude_mode,
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
) -> Annotated[PydeckLayerDefinition, Field()]:
    """
    Creates an animated trips layer definition based on the provided configuration.

    If trail_frac and/or current_frac are provided and a geodataframe is given,
    trail_length and current_time are derived from the timeline span of the
    timestamps column referenced by layer_style.get_timestamps.
    """
    layer_style = layer_style or TripsLayerStyle()
    return PydeckLayerDefinition(
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
    walk_animation: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description=ScenegraphLayerDefinition.model_fields["walk_animation"].description),
    ] = None,
    run_animation: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description=ScenegraphLayerDefinition.model_fields["run_animation"].description),
    ] = None,
    idle_animation: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description=ScenegraphLayerDefinition.model_fields["idle_animation"].description),
    ] = None,
    animation_speed: Annotated[
        float,
        AdvancedField(
            default=1.0, gt=0, le=10, description=ScenegraphLayerDefinition.model_fields["animation_speed"].description
        ),
    ] = 1.0,
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
        walk_animation=walk_animation,
        run_animation=run_animation,
        idle_animation=idle_animation,
        animation_speed=animation_speed,
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

    # results/_draw_map.py registers its own (older) myDeckWidgets build on the shared list; swap in ours.
    pdk.settings.custom_libraries = [
        lib for lib in PYDECK_CUSTOM_LIBRARIES if lib["libraryName"] != MY_DECK_WIDGETS["libraryName"]
    ] + [MY_DECK_WIDGETS]

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
            map_layers.append(pdk.Layer("TerrainLayer", **_model_dump_for_pydeck(tile_layer)))
            if tile_layer.max_zoom < max_zoom:
                max_zoom = tile_layer.max_zoom
        elif isinstance(tile_layer, BitmapLayerDefinition):
            dump = _model_dump_for_pydeck(tile_layer)
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
    elif isinstance(geo_layers, PydeckLayerDefinition):
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
                **_model_dump_for_pydeck(layer_def.layer_style),
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


def _stitch_segments(segments: gpd.GeoDataFrame, min_break_s: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate time-ordered segments into one (lon, lat) vertex array + epoch-second timestamps.

    Each segment's vertices are spaced evenly in time between its segment_start and segment_end.
    A segment that continues the previous one (starts at its last vertex and time) contributes that
    vertex once. One that doesn't follows a stretch the segment filter removed; if that stretch
    lasted more than `min_break_s` it starts a new piece (shorter ones, e.g. resting fixes the
    filter dropped, are joined up). Returns coords, times, and each vertex's piece number.
    """
    # Keep a recorded altitude (3D fixes) for drape_trips_on_terrain's absolute mode.
    has_z = bool(np.any(shapely.has_z(segments.geometry.values)))
    coords, seg_idx = shapely.get_coordinates(segments.geometry.values, include_z=has_z, return_index=True)
    counts = np.bincount(seg_idx, minlength=len(segments))
    seg_first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(len(coords)) - seg_first[seg_idx]  # vertex position within its segment
    frac = pos / np.maximum(counts[seg_idx] - 1, 1)  # single-vertex segment -> frac 0

    start = segments["segment_start"].map(pd.Timestamp.timestamp).to_numpy(dtype=np.float64)
    end = segments["segment_end"].map(pd.Timestamp.timestamp).to_numpy(dtype=np.float64)
    times = start[seg_idx] + (end - start)[seg_idx] * frac

    first = pos == 0
    joins = np.zeros(len(coords), dtype=bool)  # a segment's first vertex == the previous segment's last
    joins[1:] = first[1:] & (coords[1:] == coords[:-1]).all(axis=1) & (np.abs(times[1:] - times[:-1]) < 1)
    breaks = np.zeros(len(coords), dtype=bool)  # first vertex after a long removed stretch
    breaks[1:] = first[1:] & ~joins[1:] & (times[1:] - times[:-1] > min_break_s)
    piece = np.cumsum(breaks)
    return coords[~joins], times[~joins], piece[~joins]


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
    min_break_hours: Annotated[
        float,
        AdvancedField(
            default=12.0,
            ge=0,
            description="A stretch the trajectory segment filter removed breaks the track (no line drawn "
            "across it, marker hidden) if it lasted more than this many hours. Shorter ones, such as resting "
            "fixes the filter dropped, are joined up.",
        ),
    ] = 12.0,
    min_piece_fixes: Annotated[
        int,
        AdvancedField(
            default=3,
            ge=1,
            description="Drop pieces of track with fewer fixes than this between breaks (e.g. a collar's "
            "test fixes before deployment). If every piece is that short, only the one with the most fixes "
            "(the latest, on a tie) is kept.",
        ),
    ] = 3,
) -> AnyGeoDataFrame:
    """Stitch each group's segments into one lon/lat LineString (with the fixes' recorded altitude,
    if they have one) with per-vertex timestamps.

    Where the segment filter removed a long stretch, the track is not joined up across it: `breaks`
    lists the vertex indices j whose step to j + 1 crosses one, and the animation shows it as a data
    gap. Pipe the result through `drape_trips_on_terrain` for 3D paths over a TerrainLayer.
    """
    keep_cols = [c for c in (keep_cols or []) if c != groupby_col]

    rows = []
    for key, g in trajectory_gdf.groupby(groupby_col):
        g = g.sort_values("segment_start")
        coords, times, piece = _stitch_segments(g, min_break_s=min_break_hours * 3600)
        sizes = np.bincount(piece)
        keep = sizes[piece] >= min_piece_fixes
        if not keep.any():  # all short: keep the biggest piece, the latest on a tie
            keep = piece == len(sizes) - 1 - np.argmax(sizes[::-1])
        coords, times, piece = coords[keep], times[keep], piece[keep]
        if len(coords) < 2:  # LineString needs >= 2 vertices
            continue
        row = {groupby_col: key, **g[keep_cols].iloc[0].to_dict()}
        # Absolute epoch seconds; draw_animated_map rebases every animated layer onto one clock.
        row["timestamps"] = times.tolist()
        row["breaks"] = np.flatnonzero(piece[1:] != piece[:-1]).tolist()
        row["geometry"] = LineString(coords)
        rows.append(row)

    columns = [groupby_col, *keep_cols, "timestamps", "breaks", "geometry"]
    trips = gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs="EPSG:4326")
    return cast(AnyGeoDataFrame, trips)


@register()
def drape_trips_on_terrain(
    trips_gdf: AnyGeoDataFrame,
    terrain: Annotated[TerrainSampling, Field(description="Elevation sampling config.")] = TerrainSampling(),
) -> AnyGeoDataFrame:
    """Set every LineString vertex's z by terrain.altitude_mode: the sampled ground elevation
    (clamp_to_ground), plus terrain.offset (relative_to_ground), or the vertex's own recorded
    altitude (absolute; terrain.offset where it has none).

    All vertices across all rows are sampled in one batched call, so each DEM tile is fetched
    once. Falls back to a constant terrain.ground_elevation if sampling fails.
    """
    coords, row_idx = shapely.get_coordinates(trips_gdf.geometry.values, include_z=True, return_index=True)
    lonlats = coords[:, :2]
    if terrain.altitude_mode == "absolute":
        recorded = coords[:, 2]
        z = np.where(np.isfinite(recorded), recorded, terrain.offset)
        draped = shapely.linestrings(np.column_stack([lonlats, z]), indices=row_idx)
        return cast(
            AnyGeoDataFrame, trips_gdf.set_geometry(gpd.GeoSeries(draped, index=trips_gdf.index, crs=trips_gdf.crs))
        )
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

    lift = terrain.offset if terrain.altitude_mode == "relative_to_ground" else 0.0
    xyz = np.column_stack([lonlats, ground + lift])
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
        "color": None,  # None -> each subject's own track colour
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
                "anims": {
                    "walk": head.walk_animation,
                    "run": head.run_animation or head.walk_animation,
                    "idle": head.idle_animation,
                    "speed": head.animation_speed,
                },
            },
        )
    return spec


@register()
def animate_layer(
    layer: Annotated[PydeckLayerDefinition, Field(description="The layer to animate.", exclude=True)],
    animation: Annotated[
        LayerAnimation,
        Field(
            discriminator="kind",
            description="How the layer changes over time: a trips comet (TripsLayer only) or a time "
            "window (any layer with a time column).",
        ),
    ],
) -> Annotated[PydeckLayerDefinition, Field()]:
    """Attach an animation to a layer so draw_animated_map drives it from the shared clock."""
    if isinstance(animation, TripsAnimation) and layer.layer_type != "TripsLayer":
        raise ValueError(f"TripsAnimation needs a TripsLayer, got {layer.layer_type}.")
    base = {f.name: getattr(layer, f.name) for f in fields(PydeckLayerDefinition)}
    return AnimatedLayerDefinition(**base, animation=animation)


@register()
def draw_animated_map(
    geo_layers: Annotated[
        PydeckLayerDefinition | list[PydeckLayerDefinition] | SkipJsonSchema[None],
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
    fit_view: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Open the map framed on the animated tracks at the viewer's window size, like the "
            "video's Fit camera. The view state's pitch and bearing are kept. Off: open at the view state.",
        ),
    ] = True,
    widget_id: Annotated[str | SkipJsonSchema[None], Field(default=None, exclude=True)] = None,
) -> Annotated[str, Field()]:
    """Like draw_map, but animates layers over time from one shared clock.

    Returns a static HTML string (same contract as draw_map). Every animated layer is
    rebased onto a common timeline starting at its earliest time, so layers stay in step.
    """
    timeline = timeline or TimelineAnimation()
    geo_list = [geo_layers] if isinstance(geo_layers, PydeckLayerDefinition) else list(geo_layers or [])

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
                    "labelCol": label_col,
                    "tipCol": tip_col if tip_col in gdf.columns else "",
                    "hoverInfo": anim.hover_info,
                    # 3D models play their walk / run / idle clips by speed (dots resize slightly).
                    "speedResponse": True,
                }
            )
        else:
            gdf["__t"] = times - t0
            specs.append({"id": _layer_id(ld, i), "kind": "window", "window": anim.window_s, "fade": anim.fade_s})
        geo_list[i] = replace(ld, geodataframe=gdf)

    if view_state is None:
        view_state = view_state_from_layers(layers=geo_list, max_zoom=max_zoom)

    import pydeck as pdk  # type: ignore[import-untyped]

    # Playback runs in TimelineAnimationWidget (my-deck-widgets); it also publishes
    # window.deckInstance and window.__tripsAnim for the exporter and the Camera panel.
    timeline_widget = pdk.Widget(
        "TimelineAnimationWidget",
        id="TimelineAnimationWidget",
        placement="bottom-left",
        layer_specs=specs,
        max_time=span,
        t0=t0,
        duration_s=timeline.duration_s,
        fps_limit=timeline.fps_limit,
        auto_rotate_speed=timeline.auto_rotate_speed,
        sun={
            "on": timeline.sun_lighting,
            "night": timeline.night_brightness,
            "smoothS": timeline.day_night_smoothing_s,
        },
        pacing={
            "adaptive": timeline.adaptive_speed,
            "maxSpeedup": timeline.max_speedup,
            "restKmh": timeline.rest_speed_kmh,
            "night": timeline.fast_forward_night,
        },
        ease_s=timeline.ease_s,
        follow_smoothing_s=timeline.follow_smoothing_s,
        # 80 px padding, as the video's Fit camera (FitCamera.fit_padding).
        fit_view={"padding": 80, "maxZoom": max_zoom} if fit_view else None,
        controls=timeline.controls.model_dump(mode="json"),
    )
    extra_widgets = [timeline_widget]
    camera_on = timeline.controls.visible and timeline.controls.show_camera
    if camera_on:
        # The Camera panel (CameraPanelWidget, my-deck-widgets), built from render_animation's camera models.
        extra_widgets.append(
            pdk.Widget(
                "CameraPanelWidget", id="CameraPanelWidget", placement="top-right", cameras=camera_panel_schema()
            )
        )
    deck = _build_map_deck(
        geo_list,
        tile_layers,
        static,
        title,
        legend_style,
        max_zoom,
        view_state,
        widget_id,
        extra_widgets=extra_widgets,
    )

    html_str = deck.to_html(as_string=True)
    # The video camera (window.__cam, in the widget bundle) reads the pydeck spec from window.jsonInput.
    html_str = html_str.replace("const jsonInput =", "window.jsonInput =")
    return html_str


@register()
def create_timeline_animation(
    duration: Annotated[
        float,
        AdvancedField(
            gt=0,
            default=30.0,
            description="Playback length in seconds (e.g. 30 = the whole timeline plays in 30 seconds).",
        ),
    ] = 30.0,
    fps_limit: Annotated[float, AdvancedField(default=60.0, gt=0)] = 60.0,
    controls: Annotated[
        PlaybackControls,
        AdvancedField(default=PlaybackControls(), description="The playback bar (create_playback_controls)."),
    ] = PlaybackControls(),
) -> Annotated[TimelineAnimation, Field()]:
    """Construct the shared clock for draw_animated_map. Auto-rotate, sun lighting, adaptive speed,
    easing and follow smoothing use TimelineAnimation's defaults."""
    return TimelineAnimation(
        duration_s=duration,
        fps_limit=fps_limit,
        controls=controls,
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
    label_column: Annotated[
        str, AdvancedField(default="", description=TripsAnimation.model_fields["label_column"].description)
    ] = "",
    hover_info: Annotated[
        bool, AdvancedField(default=True, description=TripsAnimation.model_fields["hover_info"].description)
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
        label_column=label_column,
        hover_info=hover_info,
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

import os
import json
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
from ._animate import CameraConfig, camera_panel_initial, camera_panel_schema
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
        "MY_DECK_WIDGETS_URI", "https://cdn.jsdelivr.net/gh/ttemu34/my-deck-widgets@v0.21.0/dist/bundle.js"
    ),
}

TerrainStrategy = Annotated[Literal["best-available", "no-overlap", "never"], PydeckAnnotation]
# Behind and under a 3D terrain map: a dark, earthy neutral close to savanna imagery, so tiles still loading
# and the sky above a tilted horizon read as shadow rather than as white gaps.
TERRAIN_LOADING_COLOR = (58, 56, 48)
TERRAIN_BACKGROUND = "#262622"
# Basemap tiles draped over the terrain (DrapedTileLayer, my-deck-widgets). Esri World Imagery reaches
# zoom 18 (~0.6 m a pixel) across East African rangelands; zoom 19 there is mostly "not yet available".
DRAPED_IMAGERY_MAX_ZOOM = 18
# Imagery looks (GradedTerrainLayer, my-deck-widgets): colour grading of the ground only, so tracks,
# markers and labels keep their legend colours. "original" leaves the imagery as served.
ImageryLook = Literal["natural", "vivid", "warm", "original"]
IMAGERY_LOOKS: dict[str, dict[str, float]] = {
    # Esri's mosaic is a little flat and cool: lift contrast and colour slightly, warm a touch.
    "natural": {"brightness": 1.05, "contrast": 1.08, "saturation": 1.12, "warmth": 0.25},
    "vivid": {"brightness": 1.03, "contrast": 1.15, "saturation": 1.35, "warmth": 0.2},
    "warm": {"brightness": 1.03, "contrast": 1.08, "saturation": 1.15, "warmth": 0.8},
}
DEFAULT_VIGNETTE = 0.35

LABELS_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/"
    "{z}/{y}/{x}"
)


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
    get_radius: Annotated[float, AdvancedField(default=4.0, gt=0)] = 4.0
    radius_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    stroked: Annotated[bool, AdvancedField(default=True)] = True
    get_line_color: Annotated[
        tuple[int, int, int], AdvancedField(default=(255, 255, 255), json_schema_extra={"items": {"type": "integer"}})
    ] = (255, 255, 255)
    get_line_width: Annotated[float, AdvancedField(default=1.5, ge=0)] = 1.5
    line_width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    opacity: Annotated[float, AdvancedField(default=1, ge=0, le=1)] = 1


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

    Drawn as a ScenegraphLayerDefinition (to_model) with the animal's tuned orientation and size.
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
# Marker icons offered: a dot or a preset 3D animal. (ScenegraphLayerDefinition is what a 3D animal is drawn
# with; create_scenegraph_layer still builds one for code that draws its own model.)
HeadMarker = DotMarker | AnimalModelMarker


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
    preview_camera: bool = Field(
        default=True,
        description="Open with the Camera panel's preview on, so playing shows the video's camera. Moving the "
        "map turns it off. Needs the camera button.",
    )
    show_hint: bool = Field(
        default=True,
        description="Show how to pan, rotate, tilt and zoom the map, until the viewer dismisses it.",
    )
    show_map_controls: bool = Field(
        default=True,
        description="deck.gl's map buttons: zoom in and out, tilt and rotation (click to reset), back to the "
        "opening view, fullscreen, and a spinner while the map loads.",
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
            description="Marker drawn at each subject's current position: a flat dot or a 3D animal.",
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
    # deck.gl's TileLayer prop is refinementStrategy; `strategy` was silently ignored. best-available keeps
    # coarser tiles on screen while finer ones load, so zooming and panning don't open holes in the ground.
    refinement_strategy: Annotated[TerrainStrategy, AdvancedField(default="best-available")] = "best-available"
    # Ground colour where a tile's imagery hasn't arrived yet (deck.gl's default is white).
    color: Annotated[
        tuple[int, int, int],
        AdvancedField(default=TERRAIN_LOADING_COLOR, json_schema_extra={"items": {"type": "integer"}}),
    ] = TERRAIN_LOADING_COLOR
    # Mesh simplification tolerance in metres: 2 keeps ridges and gullies the 4 m default smoothed away.
    mesh_max_error: Annotated[float, AdvancedField(default=2)] = 2
    material: Annotated[bool, AdvancedField(default=True)] = True
    # Not TerrainLayer props (exclude=True keeps them out of the pydeck dump): _build_map_deck turns them
    # into DrapedTileLayers (my-deck-widgets) fitted onto this terrain.
    imagery_max_zoom: Annotated[int | SkipJsonSchema[None], Field(default=None, exclude=True)] = None
    labels: Annotated[bool, Field(default=False, exclude=True)] = False
    # Also not TerrainLayer props: _build_map_deck grades the ground with them (ImageryGradeExtension) and
    # draw_animated_map darkens the frame's edges (vignette, 0 = off).
    imagery_look: Annotated[ImageryLook, Field(default="original", exclude=True)] = "original"
    vignette: Annotated[float, Field(default=0.0, ge=0, le=1, exclude=True)] = 0.0


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


HeightAboveGround = Annotated[
    float,
    Field(default=30.0, ge=0, title="Height (m)", description="Metres above the ground the tracks are drawn at."),
]


class ClampToGround(BaseModel):
    """On the ground: tracks follow the terrain; recorded altitudes are ignored (Google's CLAMP_TO_GROUND)."""

    model_config = ConfigDict(json_schema_extra={"title": "On the ground"})
    mode: Annotated[Literal["clamp_to_ground"], Field(default="clamp_to_ground", title="Track height")] = (
        "clamp_to_ground"
    )


class RelativeToGround(BaseModel):
    """A set height above the ground, measured from elevation data (Google's RELATIVE_TO_GROUND)."""

    model_config = ConfigDict(json_schema_extra={"title": "Above the ground"})
    mode: Annotated[Literal["relative_to_ground"], Field(default="relative_to_ground", title="Track height")] = (
        "relative_to_ground"
    )
    height_m: HeightAboveGround = 30.0


class AbsoluteAltitude(BaseModel):
    """Each fix's recorded altitude above sea level (Google's ABSOLUTE). Fixes without one, as is usual for
    EarthRanger data, are drawn a set height above the ground instead of at sea level."""

    model_config = ConfigDict(json_schema_extra={"title": "Recorded altitude (where available)"})
    mode: Annotated[Literal["absolute"], Field(default="absolute", title="Track height")] = "absolute"
    fallback_height_m: Annotated[
        float,
        Field(
            default=30.0,
            ge=0,
            title="Height without an altitude (m)",
            description="Metres above the ground for fixes that have no recorded altitude.",
        ),
    ] = 30.0


class RelativeToMesh(BaseModel):
    """A set height above the 3D terrain as drawn on the map (Google's RELATIVE_TO_MESH): tracks stay on the
    rendered surface at every zoom, where elevation-data heights can sit slightly above or below it."""

    model_config = ConfigDict(json_schema_extra={"title": "Above the 3D terrain"})
    mode: Annotated[Literal["relative_to_mesh"], Field(default="relative_to_mesh", title="Track height")] = (
        "relative_to_mesh"
    )
    height_m: HeightAboveGround = 30.0


AltitudeOptions = ClampToGround | RelativeToGround | AbsoluteAltitude | RelativeToMesh
AltitudeOption = Annotated[AltitudeOptions, Field(discriminator="mode")]

# Column drape_trips_on_terrain adds in relative_to_mesh mode: draw_animated_map then places the
# tracks this many metres above the rendered terrain (deck.gl TerrainExtension, 'offset').
MESH_HEIGHT_COL = "mesh_height_m"
# Most vertices densify_m may add across all tracks; the spacing widens past it.
MAX_DENSIFIED_VERTICES = 400_000


class TerrainSampling(BaseModel):
    """How drape_trips_on_terrain sets track heights over a TerrainLayer (Google Maps 3D altitude modes).

    Skip that step for flat 2D paths.
    """

    altitude: AltitudeOption = Field(default_factory=RelativeToGround)
    densify_m: float = Field(
        default=100.0,
        ge=0,
        description="Add a point every this many metres along each track before setting heights, so lines "
        "follow the terrain between fixes instead of cutting through hills. 0 = only at the fixes.",
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

    @model_validator(mode="before")
    @classmethod
    def _legacy_altitude_mode(cls, data):
        """Settings from before the modes had their own fields: altitude_mode + offset."""
        if not isinstance(data, dict) or "altitude_mode" not in data:
            return data
        data = dict(data)
        mode, offset = data.pop("altitude_mode"), data.pop("offset", 30.0)
        if mode == "clamp_to_ground":
            data.setdefault("altitude", ClampToGround())
        elif mode == "absolute":
            data.setdefault("altitude", AbsoluteAltitude(fallback_height_m=offset))
        else:
            data.setdefault("altitude", RelativeToGround(height_m=offset))
        return data


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
    altitude: Annotated[
        AltitudeOptions,
        AdvancedField(
            default=RelativeToGround(),
            discriminator="mode",
            title="Track height",
            description="Where the tracks are drawn: on the ground, a set height above it, at each fix's "
            "recorded altitude, or a set height above the 3D terrain as drawn on the map.",
        ),
    ] = RelativeToGround(),
    densify_m: Annotated[
        float,
        AdvancedField(
            default=100.0,
            ge=0,
            title="Follow terrain every (m)",
            description="Add a point every this many metres along each track, so lines follow the terrain "
            "between fixes instead of cutting through hills. 0 = only at the fixes.",
        ),
    ] = 100.0,
) -> Annotated[TerrainSampling, Field()]:
    """Creates the track-height config consumed by drape_trips_on_terrain."""
    return TerrainSampling(
        elevation_data=elevation_data,
        elevation_decoder=elevation_decoder or dict(TERRARIUM_ELEVATION_DECODER),
        altitude=altitude,
        densify_m=densify_m,
    )


def _densify_track(xyz, times, breaks, step_m: float):
    """Points every ~step_m metres along one track, timestamps and recorded altitude interpolated.

    Steps across a data gap (`breaks`) are left alone, as no line is drawn there. Returns the new
    coordinates, timestamps (or None) and the gap indices renumbered for them.
    """
    n = len(xyz)
    if n < 2 or step_m <= 0:
        return xyz, times, breaks
    lat_mid = np.radians((xyz[:-1, 1] + xyz[1:, 1]) / 2)
    dx = np.diff(xyz[:, 0]) * 111_320.0 * np.cos(lat_mid)
    dy = np.diff(xyz[:, 1]) * 110_540.0
    k = np.maximum(np.ceil(np.hypot(dx, dy) / step_m).astype(np.int64), 1)  # sub-steps per step
    gaps = [b for b in breaks if 0 <= b < n - 1]
    k[gaps] = 1
    seg = np.repeat(np.arange(n - 1), k)
    starts = np.concatenate([[0], np.cumsum(k)])
    frac = (np.arange(len(seg)) - starts[seg]) / k[seg]
    out = np.vstack([xyz[seg] + (xyz[seg + 1] - xyz[seg]) * frac[:, None], xyz[-1:]])
    new_times = None
    if times is not None:
        t = np.asarray(times, dtype=np.float64)
        new_times = np.append(t[seg] + (t[seg + 1] - t[seg]) * frac, t[-1])
    return out, new_times, [int(starts[b]) for b in gaps]


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
    labels: Annotated[
        bool,
        Field(
            default=False,
            title="Place names",
            description="Show place names and boundaries over the basemap, draped on the terrain "
            "(Esri World Boundaries and Places).",
        ),
    ] = False,
    imagery_max_zoom: Annotated[
        int | SkipJsonSchema[None],
        AdvancedField(
            default=DRAPED_IMAGERY_MAX_ZOOM,
            ge=1,
            le=22,
            title="Basemap detail (max zoom)",
            description="Deepest zoom level of basemap tiles fetched when zooming in: 18 is about 0.6 m a pixel. "
            "Draped over the terrain, so it can be sharper than the elevation data (zoom 15). Where the basemap "
            "has nothing that deep, the terrain's own imagery shows. Blank: the terrain's zoom only.",
        ),
    ] = DRAPED_IMAGERY_MAX_ZOOM,
    imagery_look: Annotated[
        ImageryLook,
        Field(
            default="natural",
            title="Imagery look",
            description="Colour grading of the basemap: 'natural' lifts contrast and colour a little and warms "
            "it slightly, 'vivid' more so, 'warm' gives a golden-hour tone, 'original' leaves it as served. "
            "Tracks and markers keep their colours.",
        ),
    ] = "natural",
    vignette: Annotated[
        float,
        Field(
            default=DEFAULT_VIGNETTE,
            ge=0,
            le=1,
            title="Vignette",
            description="Darken the edges of the map and video slightly, drawing the eye to the middle. 0 = off.",
        ),
    ] = DEFAULT_VIGNETTE,
    mesh_max_error: Annotated[
        float,
        AdvancedField(
            default=2.0,
            gt=0,
            title="Terrain detail (m)",
            description="How far the 3D terrain may stray from the elevation data, in metres: lower keeps finer "
            "ridges and gullies but draws more triangles.",
        ),
    ] = 2.0,
) -> Annotated[TerrainLayerDefinition, Field()]:
    """Creates a terrain layer definition from elevation tiles (+ optional texture, draped place names)."""
    return TerrainLayerDefinition(
        labels=labels,
        imagery_max_zoom=imagery_max_zoom,
        imagery_look=imagery_look,
        vignette=vignette,
        mesh_max_error=mesh_max_error,
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
    """A ready-tuned 3D animal head marker for create_trips_animation."""
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
            draped = _draped_layers(pdk, tile_layer)
            props = _model_dump_for_pydeck(tile_layer)
            if draped:  # the terrain is also the surface the draped tiles are fitted onto
                # a literal: pydeck would otherwise read the "+" as an @@= accessor expression
                props["operation"] = pdk.types.String("terrain+draw")
            grade = IMAGERY_LOOKS.get(tile_layer.imagery_look)
            if grade:  # the ground only (its own texture and the imagery draped onto it), by my-deck-widgets
                map_layers.append(pdk.Layer("GradedTerrainLayer", grade=grade, **props))
            else:
                map_layers.append(pdk.Layer("TerrainLayer", **props))
            map_layers.extend(draped)
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
        # A 3D terrain needs the depth test: without it its tiles are painted in draw order, so each tile's
        # edge skirt draws over its neighbour and the cracks between tiles show (the "white gaps" between tiles).
        parameters={
            "depthTest": _has_terrain(tile_layers) or any(getattr(layer, "extruded", False) for layer in map_layers)
        },
        # A 3D terrain covers the whole map itself; a light base style underneath would only show through
        # (as white) while tiles load or above a tilted horizon. draw_animated_map darkens the page instead.
        map_style=None if _has_terrain(tile_layers) else pdk.map_styles.LIGHT_NO_LABELS,
    )


def _draped_layers(pdk, terrain: TerrainLayerDefinition) -> list:
    """Basemap tiles fitted onto the terrain at their own zoom (DrapedTileLayer, my-deck-widgets): the
    texture again, deeper than the elevation tiles go, and the place-names overlay. The TerrainLayer's own
    texture stays underneath, so the map shows imagery while these load and wherever they have none."""
    layers = []
    texture = terrain.texture if isinstance(terrain.texture, str) else None
    if texture and terrain.imagery_max_zoom and terrain.imagery_max_zoom > terrain.max_zoom:
        grade = IMAGERY_LOOKS.get(terrain.imagery_look)  # same look as the terrain's own texture
        extra = {"grade": grade} if grade else {}
        layers.append(
            pdk.Layer("DrapedTileLayer", id="draped-imagery", data=texture, max_zoom=terrain.imagery_max_zoom, **extra)
        )
    if terrain.labels:
        layers.append(
            pdk.Layer(
                "DrapedTileLayer",
                id="draped-labels",
                data=LABELS_URL,
                max_zoom=terrain.imagery_max_zoom or DRAPED_IMAGERY_MAX_ZOOM,
            )
        )
    return layers


def _map_control_widgets(pdk) -> list:
    """deck.gl's own map controls (@deck.gl/widgets): zoom; a compass whose needle follows the map's
    heading (click: north up, click again: flat); back to the opening view (TimelineAnimationWidget hands
    it the view it frames on load); fullscreen; and a spinner while tiles load. Interactive map only; the
    video hides them (render_animation's _overlay_css). GimbalWidget is not used: it drives OrbitView's
    rotationOrbit/rotationX, which a map view doesn't have, so it neither moves nor resets anything."""
    return [
        pdk.Widget("ZoomWidget", id="ZoomWidget", placement="top-left"),
        pdk.Widget("CompassWidget", id="CompassWidget", placement="top-left"),
        pdk.Widget("ResetViewWidget", id="ResetViewWidget", placement="top-left"),
        pdk.Widget("FullscreenWidget", id="FullscreenWidget", placement="top-right"),
        pdk.Widget("LoadingWidget", id="LoadingWidget", placement="top-right"),
    ]


# deck.gl's widget theme (the CSS variables @deck.gl/widgets reads), dark and translucent to sit over
# satellite imagery. my-deck-widgets' playback bar, Camera panel and hint read the same variables.
WIDGET_THEME_CSS = (
    ":root {"
    " --button-background: rgba(20, 20, 20, 0.72);"
    " --button-stroke: rgba(20, 20, 20, 0.3);"
    " --button-inner-stroke: 1px solid rgba(255, 255, 255, 0.12);"
    " --button-shadow: 0px 0px 8px 0px rgba(0, 0, 0, 0.35);"
    " --button-backdrop-filter: blur(4px);"
    " --button-icon-idle: rgba(255, 255, 255, 0.85);"
    " --button-icon-hover: #ffffff;"
    " --button-text: #ffffff;"
    " --menu-background: rgba(20, 20, 20, 0.85);"
    " --menu-backdrop-filter: blur(4px);"
    " --menu-border: 1px solid rgba(255, 255, 255, 0.15);"
    " --menu-text: #ffffff;"
    " --menu-item-hover: rgba(255, 255, 255, 0.12);"
    " color-scheme: dark; }"
)


TERRAIN_GRID_N = 96  # samples a side in the camera's line-of-sight grid
TERRAIN_GRID_PAD_M = 4000.0  # margin around the tracks, for cameras standing back from them


def _terrain_grid(tile_layers, geo_layers) -> dict | None:
    """A coarse elevation grid over the animated data (plus a margin), for the video camera's
    line-of-sight check (window.__terrainGrid): following cameras rise when a ridge would hide
    what they follow. Heights use the terrain layer's own decoder (exaggeration included), so they
    match the drawn terrain and the draped tracks. None without terrain, or if sampling fails.
    """
    terrain = next((t for t in (tile_layers or []) if hasattr(t, "elevation_data")), None)
    if terrain is None:
        return None
    bounds = [
        ld.geodataframe.to_crs("EPSG:4326").total_bounds
        for ld in geo_layers
        if getattr(ld, "geodataframe", None) is not None and len(ld.geodataframe)
    ]
    if not bounds:
        return None
    b = np.array(bounds)
    w, s, e, n = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    if not np.isfinite([w, s, e, n]).all():
        return None
    lat_mid = (s + n) / 2
    pad_lat = TERRAIN_GRID_PAD_M / 110_540.0
    pad_lon = TERRAIN_GRID_PAD_M / (111_320.0 * max(np.cos(np.radians(lat_mid)), 0.1))
    w, e = max(w - pad_lon, -180.0), min(e + pad_lon, 180.0)
    s, n = max(s - pad_lat, -85.0), min(n + pad_lat, 85.0)
    # Tiles about a quarter of the area wide: detail to spare for a 96-sample grid, few downloads.
    zoom = int(np.clip(np.floor(np.log2(1440.0 / max(e - w, 1e-6))), 5, 13))
    xs, ys = np.linspace(w, e, TERRAIN_GRID_N), np.linspace(s, n, TERRAIN_GRID_N)
    gx, gy = np.meshgrid(xs, ys)  # row-major, south to north
    try:
        z = np.asarray(
            sample_elevations(
                np.column_stack([gx.ravel(), gy.ravel()]),
                zoom=zoom,
                url=terrain.elevation_data,
                decoder=terrain.elevation_decoder,
            ),
            dtype=np.float64,
        )
    except Exception as exc:
        logger.warning("draw_animated_map: no terrain grid for the camera's line of sight (%s)", exc)
        return None
    if z.shape != (TERRAIN_GRID_N * TERRAIN_GRID_N,) or not np.isfinite(z).all():
        return None
    return {
        "w": float(w),
        "s": float(s),
        "e": float(e),
        "n": float(n),
        "nx": TERRAIN_GRID_N,
        "ny": TERRAIN_GRID_N,
        "z": np.round(z).astype(int).tolist(),
    }


def _has_terrain(tile_layers) -> bool:
    return any(hasattr(t, "elevation_data") for t in (tile_layers or []))


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
    terrain: Annotated[TerrainSampling, Field(description="Track height config.")] = TerrainSampling(),
) -> AnyGeoDataFrame:
    """Set every LineString vertex's z by terrain.altitude (Google Maps 3D altitude modes): the
    ground from elevation data (clamp_to_ground), plus a height (relative_to_ground and
    relative_to_mesh), or the vertex's own recorded altitude (absolute; a height above the ground
    where it has none).

    Tracks first get a point every terrain.densify_m metres (timestamps and `breaks` kept in step),
    so lines follow the terrain between fixes. relative_to_mesh also tags the rows (MESH_HEIGHT_COL)
    for draw_animated_map to place them above the rendered terrain; the z set here, ground + height,
    is what the camera frames.

    All vertices across all rows are sampled in one batched call, so each DEM tile is fetched
    once. Falls back to a constant terrain.ground_elevation if sampling fails.
    """
    alt = terrain.altitude
    out = trips_gdf.copy()
    has_t = "timestamps" in out.columns
    has_b = "breaks" in out.columns

    lines = []
    for i, geom in enumerate(out.geometry.values):
        xyz = shapely.get_coordinates(geom, include_z=True)
        times = out["timestamps"].iloc[i] if has_t else None
        breaks = list(out["breaks"].iloc[i]) if has_b else []
        # Not one timestamp per vertex: densifying would break the alignment, so leave the row as is.
        step = 0.0 if (times is not None and len(times) != len(xyz)) else terrain.densify_m
        lines.append(_densify_track(xyz, times, breaks, step) if step else (xyz, times, breaks))

    total = sum(len(c) for c, _, _ in lines)
    if terrain.densify_m and total > MAX_DENSIFIED_VERTICES:  # widen the spacing to stay under the cap
        wider = terrain.densify_m * total / MAX_DENSIFIED_VERTICES
        logger.info("drape_trips_on_terrain: %s points at %sm spacing; using %.0fm", total, terrain.densify_m, wider)
        return drape_trips_on_terrain(trips_gdf, terrain.model_copy(update={"densify_m": wider}))

    coords = np.vstack([c for c, _, _ in lines]) if lines else np.empty((0, 3))
    row_idx = np.repeat(np.arange(len(lines)), [len(c) for c, _, _ in lines])
    lonlats, recorded = coords[:, :2], coords[:, 2]

    ground = None
    if not (isinstance(alt, AbsoluteAltitude) and np.isfinite(recorded).all()):
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

    if isinstance(alt, ClampToGround):
        z = ground
    elif isinstance(alt, AbsoluteAltitude):
        fallback = ground + alt.fallback_height_m if ground is not None else np.zeros(len(recorded))
        z = np.where(np.isfinite(recorded), recorded, fallback)
        if not np.isfinite(recorded).any():
            logger.warning(
                "drape_trips_on_terrain: no fix has a recorded altitude; every track is drawn %sm above the ground",
                alt.fallback_height_m,
            )
    else:  # relative_to_ground, relative_to_mesh
        z = ground + alt.height_m

    draped = shapely.linestrings(np.column_stack([lonlats, z]), indices=row_idx)
    out = out.set_geometry(gpd.GeoSeries(draped, index=out.index, crs=out.crs))
    if has_t:
        out["timestamps"] = [t.tolist() if isinstance(t, np.ndarray) else t for _, t, _ in lines]
    if has_b:
        out["breaks"] = [b for _, _, b in lines]
    if isinstance(alt, RelativeToMesh):
        out[MESH_HEIGHT_COL] = float(alt.height_m)
    elif MESH_HEIGHT_COL in out.columns:
        out = out.drop(columns=MESH_HEIGHT_COL)
    return cast(AnyGeoDataFrame, out)


def _to_epoch_seconds(values: pd.Series) -> np.ndarray:
    """Datetime (naive or tz-aware) or numeric column -> float epoch seconds."""
    if not pd.api.types.is_numeric_dtype(values):
        values = pd.to_datetime(values)
    if pd.api.types.is_datetime64_any_dtype(values):
        epoch = pd.Timestamp(0, tz=values.dt.tz)
        return (values - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
    return values.to_numpy(dtype=np.float64)


def _head_spec(head: HeadMarker | ScenegraphLayerDefinition) -> dict:
    """Head-marker config for the trips animator. The dot style is always included
    because it is also the fallback when a 3D model can't be loaded."""
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
    camera: Annotated[
        CameraConfig | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="The video camera (set_video_camera). The Camera panel opens on it and previews it. "
            "None -> the panel's first camera.",
        ),
    ] = None,
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
                    # relative_to_mesh: drawn this many metres above the rendered terrain (TerrainExtension).
                    "meshHeight": (
                        float(gdf[MESH_HEIGHT_COL].iloc[0]) if MESH_HEIGHT_COL in gdf.columns and len(gdf) else None
                    ),
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
    # deck.gl's buttons go before the Camera panel, so an open panel doesn't push them down its corner.
    if timeline.controls.show_map_controls and not static:
        extra_widgets += _map_control_widgets(pdk)
    camera_on = timeline.controls.visible and timeline.controls.show_camera
    if camera_on:
        # The Camera panel (CameraPanelWidget, my-deck-widgets), built from render_animation's camera models.
        extra_widgets.append(
            pdk.Widget(
                "CameraPanelWidget",
                id="CameraPanelWidget",
                placement="top-right",
                cameras=camera_panel_schema(),
                initial=camera_panel_initial(camera),
                preview=timeline.controls.preview_camera,
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

    html_str = deck.to_html(
        as_string=True, css_background_color=TERRAIN_BACKGROUND if _has_terrain(tile_layers) else None
    )
    page_css = WIDGET_THEME_CSS
    if timeline.controls.show_map_controls and not static:
        # deck.gl's compass shows north (and turns); ecoscope's static north arrow would only repeat it.
        # The video hides the compass and brings the north arrow back (render_animation's _overlay_css).
        page_css += "\n.ecoscope-north-arrow-widget { display: none; }"
    vignette = max((getattr(t, "vignette", 0.0) for t in (tile_layers or [])), default=0.0)
    if vignette > 0:
        # A soft dark edge over the map (and so the video), under the controls; clicks pass through.
        a = round(0.55 * vignette, 3)
        page_css += (
            "\n#deck-container::after { content: ''; position: absolute; inset: 0; pointer-events: none; z-index: 1;"
            f" background: radial-gradient(ellipse at center, rgba(0,0,0,0) 55%, rgba(0,0,0,{a}) 100%); }}"
        )
    head = f"<style>{page_css}</style>"
    grid = _terrain_grid(tile_layers, geo_list)
    if grid is not None:
        head += f"<script>window.__terrainGrid = {json.dumps(grid, separators=(',', ':'))};</script>"
    html_str = html_str.replace("</head>", f"{head}\n</head>", 1)
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
    preview_camera: Annotated[
        bool,
        AdvancedField(
            default=True,
            title="Preview camera while playing",
            description=PlaybackControls.model_fields["preview_camera"].description,
        ),
    ] = True,
    show_hint: Annotated[
        bool,
        AdvancedField(
            default=True,
            title="Show map controls hint",
            description=PlaybackControls.model_fields["show_hint"].description,
        ),
    ] = True,
    show_map_controls: Annotated[
        bool,
        AdvancedField(
            default=True,
            title="Show map buttons",
            description=PlaybackControls.model_fields["show_map_controls"].description,
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
        preview_camera=preview_camera,
        show_hint=show_hint,
        show_map_controls=show_map_controls,
    )


@register()
def create_trips_animation(
    head: Annotated[
        HeadMarker,
        AdvancedField(
            default=DotMarker(),
            discriminator="marker",
            title="Marker icon",
            description="Marker drawn at each subject's current position: a flat dot or a 3D animal.",
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

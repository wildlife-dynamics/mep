import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal, get_args
import imageio_ffmpeg
from wt_registry import register
from ecoscope.platform.annotations import AdvancedField
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.json_schema import SkipJsonSchema

_browsers_ensured = False


def _use_bundled_chromium() -> None:
    """Point Playwright at a Chromium bundled in the environment (the playwright-chromium-headless-shell
    conda package installs it under <env>/share/ms-playwright), so nothing is downloaded at run time.
    Its activation script sets PLAYWRIGHT_BROWSERS_PATH; this covers launches that skip activation."""
    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        return
    bundled = Path(sys.prefix) / "share" / "ms-playwright"
    if bundled.is_dir() and any(bundled.iterdir()):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(bundled)
        # Linux: the env's fonts, or the browser draws no text where the system has no /etc/fonts.
        fonts_conf = Path(sys.prefix) / "etc" / "fonts" / "fonts.conf"
        if sys.platform.startswith("linux") and fonts_conf.exists():
            os.environ.setdefault("FONTCONFIG_FILE", str(fonts_conf))


def _ensure_playwright_browsers(force: bool = False) -> None:
    """Install Playwright Chromium binaries if not already present (once per process)."""
    global _browsers_ensured
    if _browsers_ensured and not force:
        return
    logger = logging.getLogger(__name__)
    logger.info("Ensuring Playwright Chromium browser is installed...")
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("playwright install returned non-zero: %s", result.stderr)
    else:
        _browsers_ensured = True


class DurationConfig(BaseModel):
    auto: Annotated[
        bool,
        Field(
            default=True,
            description="Match the animation's own playback length. Uncheck to set a fixed duration.",
        ),
    ] = True
    seconds: Annotated[
        float,
        Field(
            default=30.0,
            gt=0,
            title="Duration (seconds)",
            description="Video duration in seconds. Only used when 'Auto' is unchecked.",
        ),
    ] = 30.0


_RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
    "720p-vertical": (720, 1280),  # 9:16, for Reels / Shorts / WhatsApp status
    "1080p-vertical": (1080, 1920),
    "1080p-square": (1080, 1080),  # 1:1
}


class PresetResolution(BaseModel):
    """Output video resolution from a common preset."""

    model_config = ConfigDict(json_schema_extra={"title": "Preset"})
    preset: Annotated[
        Literal["720p", "1080p", "4k", "720p-vertical", "1080p-vertical", "1080p-square"],
        Field(
            default="720p",
            title="Resolution",
            description="Common output video resolution preset: landscape 16:9 (720p, 1080p, 4k), "
            "vertical 9:16 for phone stories and reels (720p-vertical, 1080p-vertical) or square 1:1 (1080p-square).",
        ),
    ] = "720p"


class CustomResolution(BaseModel):
    """Output video resolution at an exact width/height."""

    model_config = ConfigDict(json_schema_extra={"title": "Custom"})
    preset: Annotated[Literal["custom"], Field(default="custom", title="Resolution")] = "custom"
    width: Annotated[
        int,
        Field(default=1280, gt=0, title="Width", description="Custom video width in pixels."),
    ] = 1280
    height: Annotated[
        int,
        Field(default=720, gt=0, title="Height", description="Custom video height in pixels."),
    ] = 720


# Bare unions for task params: the discriminator goes on the param's own AdvancedField, because the
# compiler only reads the first FieldInfo in an Annotated and would otherwise drop the advanced flag.
ResolutionOptions = PresetResolution | CustomResolution
ResolutionConfig = Annotated[ResolutionOptions, Field(discriminator="preset")]


def _resolve_resolution(resolution: ResolutionConfig) -> tuple[int, int]:
    if isinstance(resolution, CustomResolution):
        return resolution.width, resolution.height
    return _RESOLUTION_PRESETS[resolution.preset]


# Cameras are described the Google Maps 3D way (camera-position docs): a center the camera looks at,
# its range (distance in metres), tilt, heading and lens angle (fov). The center sits on the draped
# tracks nearest it. The page converts that to deck's view state.

Range = Annotated[
    float | SkipJsonSchema[None],
    Field(
        default=None,
        gt=0,
        title="Range (m)",
        description="Camera distance from the center in metres. Leave blank to fit the data in the frame.",
    ),
]
CameraTilt = Annotated[
    float | SkipJsonSchema[None],
    Field(default=None, ge=0, le=80, description="Tilt in degrees (0 = straight down). Leave blank for the map's."),
]
CameraHeading = Annotated[
    float | SkipJsonSchema[None],
    Field(default=None, ge=-360, le=360, description="Compass heading in degrees. Leave blank for the map's."),
]
Fov = Annotated[
    float | SkipJsonSchema[None],
    AdvancedField(
        default=None,
        ge=5,
        le=80,
        title="Field of view (°)",
        description="Vertical lens angle: wider is a wide-angle look, narrower a telephoto one. Leave blank for "
        "the map's (about 37°).",
    ),
]
CenterLon = Annotated[
    float | SkipJsonSchema[None],
    Field(default=None, ge=-180, le=180, description="Center longitude. Leave blank for the center of the data."),
]
CenterLat = Annotated[
    float | SkipJsonSchema[None],
    Field(default=None, ge=-90, le=90, description="Center latitude. Leave blank for the center of the data."),
]


class FixedCamera(BaseModel):
    """One view for the whole clip: by default the whole scene, at the map's tilt and heading."""

    model_config = ConfigDict(json_schema_extra={"title": "Fixed view"})
    type_: Annotated[Literal["fixed"], Field(default="fixed", title="Camera")] = "fixed"
    lon: CenterLon = None
    lat: CenterLat = None
    range_m: Range = None
    tilt: CameraTilt = None
    heading: CameraHeading = None
    fov: Fov = None


class FlyAroundCamera(BaseModel):
    """Fly to a point, then circle it -- the Google Maps 3D flyCameraTo + flyCameraAround pair.

    The fly-to follows deck's FlyToInterpolator arc (out, across and back in) from the whole-scene
    view; the orbit then turns at a constant rate for `rounds` turns.
    """

    model_config = ConfigDict(json_schema_extra={"title": "Fly to & around"})
    type_: Annotated[Literal["fly_around"], Field(default="fly_around", title="Camera")] = "fly_around"
    lon: CenterLon = None
    lat: CenterLat = None
    range_m: Range = None
    tilt: CameraTilt = None
    heading: Annotated[
        float | SkipJsonSchema[None],
        Field(default=None, ge=-360, le=360, description="Compass heading on arrival. Leave blank for the map's."),
    ] = None
    fov: Fov = None
    fly_frac: Annotated[
        float,
        Field(
            default=0.25,
            ge=0,
            le=1,
            description="Share of the clip spent flying in from the whole-scene view. 0 = start at the center.",
        ),
    ] = 0.25
    rounds: Annotated[
        float, Field(default=1.0, ge=0, description="Full turns around the center after arriving. 0 = hold.")
    ] = 1.0


class FollowSubjectCamera(BaseModel):
    """Stays on one subject at a set range and tilt, behind it and facing where it is heading, lagging a
    little behind its moves."""

    model_config = ConfigDict(json_schema_extra={"title": "Follow a subject"})
    type_: Annotated[Literal["chase"], Field(default="chase", title="Camera")] = "chase"
    subject: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            description="Value identifying the subject in any column (e.g. its name or group id). "
            "Leave blank for the longest-running track.",
        ),
    ] = None
    range_m: Annotated[
        float, Field(default=800.0, gt=0, title="Range (m)", description="Camera distance from the subject in metres.")
    ] = 800.0
    tilt: Annotated[float, Field(default=60.0, ge=0, le=80, description="Tilt in degrees (0 = straight down).")] = 60.0
    heading: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            ge=-360,
            le=360,
            description="Compass heading in degrees. Leave blank to face the subject's direction of travel.",
        ),
    ] = None
    fov: Fov = None
    follow_smoothing: Annotated[
        float,
        AdvancedField(
            default=0.3,
            ge=0,
            le=1,
            description="Share of the gap to the subject closed every 1/30 s: low = drifts behind, 1 = locked on.",
        ),
    ] = 0.3
    turn_smoothing: Annotated[
        float,
        AdvancedField(
            default=0.15,
            ge=0,
            le=1,
            description="Share of the turn towards the direction of travel made every 1/30 s: low = slow, "
            "sweeping turns, 1 = snaps.",
        ),
    ] = 0.15


class FollowActionCamera(BaseModel):
    """Keeps the moving subjects in frame: frames what animated in the last `follow_window` of the
    timeline, re-fitting smoothly as they move and spread."""

    model_config = ConfigDict(json_schema_extra={"title": "Follow the action"})
    type_: Annotated[Literal["follow"], Field(default="follow", title="Camera")] = "follow"
    subjects: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Only frame these subjects (their value in any column, e.g. names), zooming out as they "
            "spread apart. Leave blank for everything that animates.",
        ),
    ] = None
    tilt: CameraTilt = None
    heading: CameraHeading = None
    face_travel: Annotated[
        bool, AdvancedField(default=False, description="Turn the camera to face the direction of travel.")
    ] = False
    fov: Fov = None
    follow_window: Annotated[
        float,
        AdvancedField(default=0.1, gt=0, le=1, description="Share of the timeline treated as 'recent' when framing."),
    ] = 0.1
    follow_smoothing: Annotated[
        float,
        AdvancedField(
            default=0.25,
            ge=0,
            le=1,
            description="Share of the gap to the action closed every 1/30 s: low = smooth but laggy, 1 = snaps to "
            "it. Independent of fps.",
        ),
    ] = 0.25
    fit_padding: Annotated[int, AdvancedField(default=80, ge=0, description="Padding in pixels around the action.")] = (
        80
    )


class CameraKeyframe(BaseModel):
    """One waypoint on a keyframes camera path.

    Only ``lon``/``lat`` are required. ``t`` places the keyframe on the clip timeline (0 = first
    captured frame, 1 = last); when omitted on every keyframe they are spaced evenly.
    ``zoom``/``pitch``/``bearing`` left as None are interpolated between the nearest keyframes that
    define them, or fall back to the whole-scene framing (zoom) and the map's pitch/bearing.
    """

    lon: Annotated[float, Field(description="Longitude of the camera look-at point.")]
    lat: Annotated[float, Field(description="Latitude of the camera look-at point.")]
    t: Annotated[
        float | None,
        Field(
            default=None,
            description="Position on the clip timeline (0 = start, 1 = end). Any increasing numbers work (they "
            "are normalized to 0–1); leave blank on every keyframe to space them evenly.",
        ),
    ] = None
    zoom: Annotated[
        float | None,
        Field(
            default=None,
            description="Zoom level at this keyframe. Leave blank to blend from neighbouring keyframes, or fit the "
            "whole scene if none set it.",
        ),
    ] = None
    pitch: Annotated[
        float | None,
        Field(
            default=None,
            description="Camera tilt in degrees at this keyframe. Leave blank to blend from neighbouring keyframes, "
            "or use the map's tilt if none set it.",
        ),
    ] = None
    bearing: Annotated[
        float | None,
        Field(
            default=None,
            description="Camera heading in degrees at this keyframe. Leave blank to blend from neighbouring "
            "keyframes, or use the map's heading if none set it.",
        ),
    ] = None


_KF_CHANNELS = ("zoom", "pitch", "bearing")


def _load_keyframes_file(path: str) -> list[dict]:
    """Load camera keyframes from a .json (list of objects), .geojson (Point features; t/zoom/pitch/
    bearing read from properties), or .csv/.tsv (lon/lat columns, optional t/zoom/pitch/bearing) file."""
    import csv

    p = Path(remove_file_scheme(path))
    suffix = p.suffix.lower()

    def norm(d: dict) -> dict:
        out: dict = {}
        for key, aliases in (
            ("lon", ("lon", "longitude", "lng", "x")),
            ("lat", ("lat", "latitude", "y")),
            ("t", ("t", "time", "frac")),
            ("zoom", ("zoom",)),
            ("pitch", ("pitch", "tilt")),
            ("bearing", ("bearing", "heading")),
        ):
            for a in aliases:
                if a in d and d[a] not in (None, ""):
                    out[key] = float(d[a])
                    break
        return out

    if suffix in (".json", ".geojson"):
        data = json.loads(p.read_text())
        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            rows = []
            for f in data.get("features", []):
                g = f.get("geometry") or {}
                if g.get("type") != "Point":
                    continue
                d = {str(k).lower(): v for k, v in (f.get("properties") or {}).items()}
                d["lon"], d["lat"] = g["coordinates"][:2]
                rows.append(norm(d))
            return rows
        if isinstance(data, list):
            return [norm({str(k).lower(): v for k, v in d.items()}) for d in data]
        raise ValueError(f"Unsupported keyframe JSON structure in {path}")
    if suffix in (".csv", ".tsv"):
        with open(p, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ",")
            return [norm({(k or "").strip().lower(): v for k, v in row.items()}) for row in reader]
    raise ValueError(f"Unsupported keyframe file type: {path} (use .json, .geojson, .csv, or .tsv)")


def _resolve_keyframes(raw: list[dict]) -> list[dict]:
    """Turn sparse user keyframes into a complete, sorted camera path.

    - Missing ``t`` -> even spacing; any increasing times are normalized to 0–1.
    - Missing zoom/pitch/bearing -> interpolated between the nearest keyframes that define them
      (held flat past the ends). A channel no keyframe defines stays None for the page to fill
      with the whole-scene zoom and the map's pitch/bearing.
    - Bearings are unwrapped so the page's interpolation always turns the short way.
    """
    kf = [dict(k) for k in raw]
    if len(kf) < 2:
        raise ValueError("camera='keyframes' needs at least 2 keyframes")
    for k in kf:
        if k.get("lon") is None or k.get("lat") is None:
            raise ValueError(f"keyframe is missing lon/lat: {k}")
    n = len(kf)
    if any(k.get("t") is None for k in kf):
        for i, k in enumerate(kf):
            k["t"] = i / (n - 1)
    kf.sort(key=lambda k: k["t"])
    t0, t1 = kf[0]["t"], kf[-1]["t"]
    if t1 > t0:
        for k in kf:
            k["t"] = (k["t"] - t0) / (t1 - t0)
    for ch in _KF_CHANNELS:
        idxs = [i for i, k in enumerate(kf) if k.get(ch) is not None]
        if not idxs:
            for k in kf:
                k[ch] = None
            continue
        if ch == "bearing":  # unwrap defined values -> shortest-path rotation
            for a, b in zip(idxs, idxs[1:]):
                d = (kf[b][ch] - kf[a][ch] + 180) % 360 - 180
                kf[b][ch] = kf[a][ch] + d
        for i in range(idxs[0]):  # hold flat before the first defined value
            kf[i][ch] = kf[idxs[0]][ch]
        for i in range(idxs[-1] + 1, n):  # ...and after the last
            kf[i][ch] = kf[idxs[-1]][ch]
        for a, b in zip(idxs, idxs[1:]):  # linear fill between defined values
            span_t = max(kf[b]["t"] - kf[a]["t"], 1e-9)
            for i in range(a + 1, b):
                fr = (kf[i]["t"] - kf[a]["t"]) / span_t
                kf[i][ch] = kf[a][ch] + (kf[b][ch] - kf[a][ch]) * fr
    return [{"t": k["t"], "lon": k["lon"], "lat": k["lat"], **{ch: k[ch] for ch in _KF_CHANNELS}} for k in kf]


class KeyframesFromFile(BaseModel):
    """Camera path loaded from an uploaded waypoint file."""

    model_config = ConfigDict(json_schema_extra={"title": "Upload waypoint file"})
    type_: Annotated[Literal["file"], Field(default="file", title="Source")] = "file"
    keyframes_file: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            description="Path to an uploaded keyframe file: a .json list of {lon, lat, t?, zoom?, pitch?, bearing?} "
            "objects, a .geojson of Point features (extras read from properties), or a .csv/.tsv with lon/lat "
            "columns. Required for this source.",
        ),
    ] = None


class KeyframesFromSubject(BaseModel):
    """Camera path auto-derived by following one subject through the animated data.

    The subject is matched against every column, so it works for trips and for point or
    polygon layers that carry the subject's column.
    """

    model_config = ConfigDict(json_schema_extra={"title": "Follow a subject"})
    type_: Annotated[Literal["subject"], Field(default="subject", title="Source")] = "subject"
    subject: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            description="Value identifying the subject in any column (e.g. its name or group id). "
            "Leave blank to follow the longest-running track.",
        ),
    ] = None


KeyframeSource = Annotated[KeyframesFromSubject | KeyframesFromFile, Field(discriminator="type_")]


class KeyframesCamera(BaseModel):
    """Flies through waypoints while the data animates: add views in the Camera panel, upload a
    waypoint file, or follow a subject.

    Waypoints given in ``keyframes`` take priority over ``source``. Waypoints without a zoom use
    the whole-scene framing.
    """

    model_config = ConfigDict(json_schema_extra={"title": "Keyframes"})
    type_: Annotated[Literal["keyframes"], Field(default="keyframes", title="Camera")] = "keyframes"
    keyframes: Annotated[
        list[CameraKeyframe] | SkipJsonSchema[None],
        Field(
            default=None,
            description="Camera waypoints, flown through in order. Each needs lon/lat; t (0–1 clip position), "
            "zoom, pitch and bearing are optional. Leave empty to build the path from Source instead.",
        ),
    ] = None
    source: Annotated[
        KeyframeSource,
        AdvancedField(
            default=KeyframesFromSubject(),
            description="How to build the path when `keyframes` is empty: upload a file, or follow a subject.",
        ),
    ] = KeyframesFromSubject()
    keyframe_easing: Annotated[
        Literal["smooth", "linear", "spline"],
        AdvancedField(
            default="smooth",
            description="How the camera moves between keyframes: 'smooth' eases in/out of each waypoint, 'linear' "
            "moves at constant speed, 'spline' curves through waypoints (Catmull-Rom) without pausing at them.",
        ),
    ] = "smooth"
    fov: Fov = None


def _raw_keyframes(camera: "CameraOptions") -> list[dict]:
    """A keyframes camera's waypoints, given inline or via file. An empty list means 'follow
    `source`'s subject'; checked before the browser launches."""
    if not isinstance(camera, KeyframesCamera):
        return []
    raw = [k.model_dump() for k in (camera.keyframes or [])]
    if not raw and isinstance(camera.source, KeyframesFromFile):
        if not camera.source.keyframes_file:
            raise ValueError("camera='keyframes' with an uploaded file source needs keyframes_file")
        raw = _load_keyframes_file(camera.source.keyframes_file)
    if len(raw) == 1:
        raise ValueError(
            "camera='keyframes' needs at least 2 keyframes, got 1. "
            "Add more keyframes, or leave the list empty to follow a subject."
        )
    return raw


CameraOptions = FixedCamera | FlyAroundCamera | FollowSubjectCamera | FollowActionCamera | KeyframesCamera
CameraConfig = Annotated[CameraOptions, Field(discriminator="type_")]
_CAMERA_ADAPTER: TypeAdapter = TypeAdapter(CameraConfig)

# Fields the animated map's Camera panel edits with its own controls rather than a form input.
_PANEL_CUSTOM_FIELDS = ("type_", "keyframes", "source")


def camera_panel_schema() -> list[dict]:
    """Form description of every camera, for the Camera panel in draw_animated_map's page.

    Built from the camera models so the panel offers exactly what render_animation
    accepts, with the same defaults, limits and help text.
    """
    cameras = []
    for cls in get_args(CameraOptions):
        props = cls.model_json_schema()["properties"]
        fields = [
            {
                "name": name,
                "title": prop.get("title", name),
                "description": prop.get("description", ""),
                "type": "enum" if "enum" in prop else prop.get("type", "string"),
                "options": prop.get("enum"),
                "default": prop.get("default"),
                # Optional (`| SkipJsonSchema[None]`) fields keep their limits as raw ge/le/gt/lt.
                "min": next((prop[k] for k in ("minimum", "exclusiveMinimum", "ge", "gt") if k in prop), None),
                "max": next((prop[k] for k in ("maximum", "exclusiveMaximum", "le", "lt") if k in prop), None),
            }
            for name, prop in props.items()
            if name not in _PANEL_CUSTOM_FIELDS
        ]
        cameras.append(
            {
                "type": cls.model_fields["type_"].default,
                "title": cls.model_config.get("json_schema_extra", {}).get("title", cls.__name__),
                "description": " ".join((cls.__doc__ or "").split("\n\n")[0].split()),
                "fields": fields,
            }
        )
    return cameras


def parse_camera_settings(settings: str) -> CameraOptions:
    """Camera designed in the animated map's Camera panel: the copied settings text, or
    the path of the downloaded settings file."""
    text = settings.strip()
    if not text.startswith("{"):
        text = Path(remove_file_scheme(text)).read_text(encoding="utf-8")
    try:
        return _CAMERA_ADAPTER.validate_python(json.loads(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"Camera settings are not valid JSON: {e}") from e


def _launch_args(gl: str):
    base = [
        "--headless=new",
        "--ignore-gpu-blocklist",
        "--enable-unsafe-swapchains",
        "--no-sandbox",
        "--hide-scrollbars",
    ]
    if gl == "software":
        return base + ["--use-gl=angle", "--use-angle=swiftshader"]
    if gl in ("angle", "auto"):
        return base + ["--use-gl=angle"]  # ANGLE picks Metal/GL/Vulkan -> real GPU
    return base


class VideoOverlays(BaseModel):
    """What the video shows over the map. The playback bar and camera panel are never shown."""

    clock: bool = Field(
        default=True, description="The data date and time in a corner, formatted as on the map's playback bar."
    )
    clock_position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = Field(
        default="top-right", description="Corner for the date and time."
    )
    legend: bool = Field(default=True, description="The legend.")
    north_arrow: bool = Field(default=True, description="The north arrow.")
    scale_bar: bool = Field(default=True, description="The scale bar.")
    title: bool = Field(default=True, description="The map title, if it has one.")
    labels: bool = Field(default=True, description="Labels above the markers (e.g. subject names).")


# The map's widgets by their rendered classes (the deck.gl widgets carry no usable ids).
_OVERLAY_SELECTORS = {
    "legend": ".ecoscope-legend-widget",
    "north_arrow": ".ecoscope-north-arrow-widget",
    "scale_bar": ".ecoscope-scale-widget",
    "title": ".ecoscope-title-widget",
}


def _overlay_css(overlays: VideoOverlays, height: int) -> str:
    """Stylesheet for the video: interactive-only controls and switched-off widgets hidden, the clock placed."""
    hidden = [".deck-widget-save-image", "#timeline-controls", "#camera-panel", ".camera-panel-widget"]
    hidden += [sel for name, sel in _OVERLAY_SELECTORS.items() if not getattr(overlays, name)]
    vert, horiz = overlays.clock_position.split("-")
    size = max(12, round(height * 0.026))  # ~19 px at 720p, ~28 px at 1080p, scaled for vertical video too
    margin = max(10, round(height * 0.02))
    return (
        ", ".join(hidden) + " { display: none !important; }\n"
        f"#video-clock {{ position: fixed; {vert}: {margin}px; {horiz}: {margin}px; z-index: 20;"
        f" padding: {round(size * 0.3)}px {round(size * 0.55)}px; border-radius: {round(size * 0.35)}px;"
        f" background: rgba(20, 20, 20, 0.72); color: #fff; font: 600 {size}px/1.2 system-ui, sans-serif;"
        " font-variant-numeric: tabular-nums; white-space: nowrap; pointer-events: none; }"
    )


async def _prepare_page(
    browser, html_uri, *, width, height, device_scale_factor, head_ready_timeout_ms, overlays, map_timeout_ms
):
    """Open a page, load the scene, take control of the autoplay loop, inject helper and the
    video overlays, and wait for the base map. Returns (page, pending_counter_dict)."""
    page = await browser.new_page(
        viewport={"width": width, "height": height},
        device_scale_factor=device_scale_factor,
    )
    # Count in-flight tile requests so we can wait for the basemap to paint. Tiles arrive as images
    # or, for deck's tile/terrain layers and vector base maps, through fetch/XHR.
    pending = {"n": 0}
    tile_types = {"image", "fetch", "xhr"}

    def _on_req(req):
        if req.resource_type in tile_types:
            pending["n"] += 1

    def _on_done(req):
        if req.resource_type in tile_types:
            pending["n"] = max(0, pending["n"] - 1)

    page.on("request", _on_req)
    page.on("requestfinished", _on_done)
    page.on("requestfailed", _on_done)

    # Not "load": one slow third-party resource would hold it up. Readiness is checked below.
    await page.goto(html_uri, wait_until="domcontentloaded", timeout=map_timeout_ms)
    await page.wait_for_function("() => window.__tripsAnim && window.__tripsAnim.ready", timeout=map_timeout_ms)
    await page.evaluate("() => window.__tripsAnim.pause()")
    await page.wait_for_function("() => window.__tripsAnim.headReady", timeout=head_ready_timeout_ms)
    # The camera (window.__cam) ships in the widget bundle (my-deck-widgets >= 0.17).
    if not await page.evaluate("() => !!(window.__cam && window.__cam.path)"):
        raise RuntimeError(
            "The animated map has no video camera (window.__cam). Redraw it with draw_animated_map so it loads "
            "my-deck-widgets 0.17 or later."
        )
    await page.add_style_tag(content=_overlay_css(overlays, height))
    await page.evaluate(
        """([clock, labels]) => {
            const a = window.__tripsAnim;
            if (!labels && typeof a.setLabels === 'function') a.setLabels(false);
            // The clock needs the widget's formatter (my-deck-widgets >= 0.15); older pages go without.
            if (clock && typeof a.formatTime === 'function') {
              const el = document.createElement('div');
              el.id = 'video-clock';
              document.body.appendChild(el);
            }
        }""",
        [overlays.clock, overlays.labels],
    )
    # The whole base map in, before the first frame (later frames wait per frame).
    await _await_ready(page, pending, settle_timeout_ms=map_timeout_ms, settle_ms=0)
    return page, pending


# Every deck layer loaded (terrain/tile layers: the tiles in view) and the base map's tiles in.
_TILES_LOADED_JS = """() => {
    const a = window.__tripsAnim;
    if (a && typeof a.mapLoaded === 'function') return a.mapLoaded();
    const d = window.deckInstance; if (!d) return true;
    try {
      const lm = d.layerManager || (d.deck && d.deck.layerManager);
      const ls = (lm && lm.getLayers) ? lm.getLayers() : (d.props.layers || []);
      return ls.every(l => l.isLoaded !== false);
    } catch (e) { return true; }
}"""


async def _await_ready(page, pending, *, settle_timeout_ms, settle_ms, stable_ms=60):
    """Wait until no tile/image requests are in flight AND deck reports loaded.
    Falls through after settle_timeout_ms so a single slow tile can't stall us."""
    deadline = time.time() + settle_timeout_ms / 1000.0
    while True:
        if pending["n"] <= 0 and await page.evaluate(_TILES_LOADED_JS):
            await page.wait_for_timeout(stable_ms)  # confirm it stays settled
            if pending["n"] <= 0:
                break
        if time.time() >= deadline:
            break
        await page.wait_for_timeout(15)
    if settle_ms:
        await page.wait_for_timeout(settle_ms)  # compositor cushion


def _frame_path(frame_dir: str, idx: int, ext: str) -> str:
    return os.path.join(frame_dir, f"m_{idx:06d}.{ext}")


async def _render_frames(page, pending, frames, clip, quality, capture, frame_dir, ext, progress):
    """Render an iterable of (index, t, viewState) to numbered files. Each file is written under a
    temporary name and renamed when complete, so a resumed render never trusts a half-written frame."""
    shot_kwargs = {"clip": clip, "type": quality.capture_format}
    if quality.capture_format == "jpeg":
        shot_kwargs["quality"] = quality.jpeg_quality
    for idx, t, vs in frames:
        await page.evaluate(
            """([t, vs]) => {
                const d = window.deckInstance, a = window.__tripsAnim;
                if (vs && d) d.setProps({ viewState: vs });
                a.renderAt(t);
                const clock = document.getElementById('video-clock');
                if (clock) clock.textContent = a.formatTime(t);
                try { d.redraw && d.redraw('export'); } catch (e) {}
            }""",
            [t, vs],
        )
        # let deck issue tile requests for the new viewport, then drain them
        await page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        await _await_ready(page, pending, settle_timeout_ms=capture.settle_timeout_ms, settle_ms=capture.settle_ms)
        final = _frame_path(frame_dir, idx, ext)
        shot_kwargs["path"] = final + ".part"
        await page.screenshot(**shot_kwargs)
        os.replace(final + ".part", final)
        progress(idx)


class VideoQuality(BaseModel):
    """Image capture and H.264 encoding settings."""

    capture_format: Literal["jpeg", "png"] = Field(
        default="jpeg", description="Per-frame image format: 'jpeg' (fast, small) or 'png' (lossless, larger)."
    )
    jpeg_quality: int = Field(default=92, ge=1, le=100, description="JPEG quality (1–100). Only used for 'jpeg'.")
    crf: int = Field(
        default=18, ge=0, le=51, description="H.264 constant rate factor (0 = lossless, 51 = worst). Lower = better."
    )
    x264_preset: Literal[
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"
    ] = Field(default="veryfast", description="x264 encoding speed preset (ultrafast → veryslow).")
    device_scale_factor: int = Field(default=1, gt=0, description="Browser device pixel ratio. 2 = HiDPI output.")


class CaptureOptions(BaseModel):
    """Headless-browser settings for frame capture."""

    gl: Literal["auto", "angle", "software"] = Field(
        default="auto", description="WebGL backend: 'auto'/'angle' use the GPU; 'software' only if there is none."
    )
    workers: Annotated[int, Field(gt=0)] | Literal["auto"] = Field(
        default="auto",
        description="Parallel browser pages capturing frames. 'auto' sizes it from CPU cores, free memory "
        "and the number of frames, leaving headroom for the rest of the machine.",
    )
    settle_ms: int = Field(default=30, ge=0, description="Extra ms to wait after tiles load, before each frame.")
    settle_timeout_ms: int = Field(default=8000, gt=0, description="Max ms to wait for tiles per frame.")
    head_ready_timeout_ms: int = Field(
        default=30000, gt=0, description="Max ms to wait for lazily loaded pieces (3D head model, filters)."
    )
    map_timeout_ms: int = Field(
        default=60000,
        gt=0,
        description="Max ms to wait for the whole base map and terrain to load before the first frame.",
    )
    resume: bool = Field(
        default=True,
        description="Keep captured frames next to the video until it is written, so a failed or interrupted "
        "render with the same settings picks up where it stopped instead of starting over.",
    )


def _camera_opts(camera: CameraOptions, width: int, height: int, fps: int = 30) -> dict:
    """Options for window.__cam.path: the camera's fields (the page fills its own defaults), a
    keyframes camera's resolved waypoints, and the video size the framing is fitted to."""
    fields = camera.model_dump()
    preset = fields.pop("type_")
    if isinstance(camera, KeyframesCamera):
        raw = _raw_keyframes(camera)
        fields.pop("source")
        fields["keyframes"] = _resolve_keyframes(raw) if raw else None
        # No waypoints -> the page derives them from this subject's track.
        fields["subject"] = camera.source.subject if isinstance(camera.source, KeyframesFromSubject) else None
    return {**fields, "preset": preset, "width": width, "height": height, "fps": fps}


# Rough per-page costs for the automatic worker count.
_PAGE_MEMORY_BYTES = 600 * 1024**2  # Chromium page with WebGL, tiles and frame buffers
_MIN_FRAMES_PER_PAGE = 40  # below this a page's load time outweighs the frames it captures
_MAX_GPU_PAGES = 6  # pages share one GPU; past this they just queue on it


def _auto_workers(n_frames: int, gl: str) -> int:
    """Pages to capture with: half the usable cores (software GL burns about a core per
    page), capped by free memory, GPU sharing and how many frames there are to split."""
    try:
        cores = len(os.sched_getaffinity(0))  # respects container / taskset limits (Linux)
    except AttributeError:
        cores = os.cpu_count() or 2
    limit = max(1, cores // 2)
    if gl != "software":
        limit = min(limit, _MAX_GPU_PAGES)
    try:
        import psutil

        limit = min(limit, max(1, psutil.virtual_memory().available // _PAGE_MEMORY_BYTES))
    except ImportError:
        pass
    return max(1, min(limit, n_frames // _MIN_FRAMES_PER_PAGE))


async def _launch_browser(p, gl: str):
    try:
        return await p.chromium.launch(headless=True, args=_launch_args(gl))
    except Exception as e:
        if "Executable doesn't exist" not in str(e):
            raise
        _ensure_playwright_browsers(force=True)
        return await p.chromium.launch(headless=True, args=_launch_args(gl))


def _assemble_sequence(frame_dir: str, ext: str, n_main: int, n_hold_start: int, n_hold_end: int) -> str:
    """The encoder's numbered input: the first frame held n_hold_start times, every captured frame,
    then the last frame held n_hold_end times. Hard links, so holds cost no disk or capture time."""
    seq_dir = os.path.join(frame_dir, "sequence")
    shutil.rmtree(seq_dir, ignore_errors=True)
    os.makedirs(seq_dir)
    order = [0] * n_hold_start + list(range(n_main)) + [n_main - 1] * n_hold_end
    for i, src in enumerate(order):
        src_path, dst_path = _frame_path(frame_dir, src, ext), os.path.join(seq_dir, f"f_{i:06d}.{ext}")
        try:
            os.link(src_path, dst_path)
        except OSError:  # no hard links on this file system
            shutil.copyfile(src_path, dst_path)
    return seq_dir


def _frames_dir(out_path: str, key: str) -> str:
    """Where a resumable render keeps its frames: beside the video, named by the render's settings,
    or in the temp dir if the video's folder isn't writable."""
    stem = Path(out_path).stem
    for parent in (Path(out_path).resolve().parent, Path(tempfile.gettempdir())):
        d = parent / f".{stem}-frames-{key}"
        try:
            d.mkdir(parents=True, exist_ok=True)
            return str(d)
        except OSError:
            continue
    return tempfile.mkdtemp(prefix="anim_frames_")


def _render_key(html_path: Path, settings: dict) -> str:
    """Identifies a render: the page's content and every setting that changes a captured frame."""
    h = hashlib.sha256(html_path.read_bytes())
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _encode_mp4(frame_dir: str, ext: str, fps: int, quality: VideoQuality, out_path: str) -> None:
    """Assemble the numbered frame sequence into H.264."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", os.path.join(frame_dir, f"f_%06d.{ext}"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", quality.x264_preset,
        "-pix_fmt", "yuv420p",
        "-crf", str(quality.crf),
        "-movflags", "+faststart",
        out_path,
    ]  # fmt: skip
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + proc.stderr.decode("utf-8", "ignore")[-2000:])


async def render_animation_async(
    html_path: str,
    output_dir: str | None = None,
    out_path: str = "animation.mp4",
    camera: CameraOptions = FixedCamera(),
    camera_settings: str | None = None,
    fps: int = 30,
    duration: DurationConfig = DurationConfig(),
    resolution: ResolutionOptions = PresetResolution(),
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    hold_start_s: float = 1.0,
    hold_end_s: float = 2.0,
    overlays: VideoOverlays = VideoOverlays(),
    quality: VideoQuality = VideoQuality(),
    capture: CaptureOptions = CaptureOptions(),
    verbose: bool = True,
) -> str:
    """Async core of render_animation. ``await`` this directly inside a notebook if you prefer."""
    if camera_settings:
        camera = parse_camera_settings(camera_settings)
    keyframes = _raw_keyframes(camera)  # a bad waypoint file fails here, before the browser launches
    width, height = _resolve_resolution(resolution)
    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(html_path)
    if output_dir:
        out_path = os.path.join(output_dir, os.path.basename(str(out_path)))
    ext = "jpg" if quality.capture_format == "jpeg" else "png"
    t_wall = time.time()

    def log(*a):
        if verbose:
            print("[render]", *a, file=sys.stderr, flush=True)

    async def open_page():
        return await _prepare_page(
            browser,
            html_path.as_uri(),
            width=width,
            height=height,
            device_scale_factor=quality.device_scale_factor,
            head_ready_timeout_ms=capture.head_ready_timeout_ms,
            overlays=overlays,
            map_timeout_ms=capture.map_timeout_ms,
        )

    # Everything that changes a captured frame; a resumed render must match it exactly.
    key = _render_key(
        html_path,
        {
            "camera": camera.model_dump(mode="json"),
            "keyframes": keyframes,  # a waypoint file's contents, not just its path
            "fps": fps,
            "duration": duration.model_dump(),
            "size": [width, height],
            "range": [start_frac, end_frac],
            "overlays": overlays.model_dump(),
            "frames": quality.model_dump(include={"capture_format", "jpeg_quality", "device_scale_factor"}),
        },
    )
    frame_dir = _frames_dir(out_path, key) if capture.resume else tempfile.mkdtemp(prefix="anim_frames_")

    _use_bundled_chromium()
    try:
        async with async_playwright() as p:
            browser = await _launch_browser(p, capture.gl)

            # One page reads the timeline and base view and computes the whole camera path.
            page0, pending0 = await open_page()
            span = await page0.evaluate("() => window.__cam.span() || window.__tripsAnim.span")
            natural = await page0.evaluate("() => window.__tripsAnim.durationSec || 0")
            seconds = float(natural) if duration.auto and natural > 0 else float(duration.seconds)
            n_frames = max(1, int(round(fps * seconds)))

            t_lo, t_hi = span * start_frac, span * end_frac
            # Frames are evenly spaced in playback time. With adaptive speed the widget maps playback
            # progress to data time (timeAt), so quiet stretches go by faster, as on the map.
            times = await page0.evaluate(
                """([lo, hi, n]) => {
                  const a = window.__tripsAnim, at = (i) => (n > 1 ? i / (n - 1) : 1);
                  if (typeof a.timeAt !== 'function') {
                    return Array.from({ length: n }, (_, i) => lo + (hi - lo) * at(i));
                  }
                  const p0 = a.progressAt(lo), p1 = a.progressAt(hi);
                  return Array.from({ length: n }, (_, i) => a.timeAt(p0 + (p1 - p0) * at(i)));
                }""",
                [t_lo, t_hi, n_frames],
            )
            opts = _camera_opts(camera, width, height, fps)
            views = await page0.evaluate("([times, opts]) => window.__cam.path(times, opts)", [times, opts])

            canvas = await page0.query_selector("#deck-container canvas")
            if canvas is None:
                raise RuntimeError("deck-container canvas not found")
            clip = await canvas.bounding_box()

            # Frames a previous, interrupted render with the same settings already captured.
            frames = [
                f for f in zip(range(n_frames), times, views) if not os.path.exists(_frame_path(frame_dir, f[0], ext))
            ]
            todo = len(frames)
            workers = _auto_workers(todo, capture.gl) if capture.workers == "auto" else capture.workers
            log(
                f"span={span} duration={seconds:.2f}s frames={n_frames} camera={camera.type_} "
                f"gl={capture.gl} workers={workers}{' (auto)' if capture.workers == 'auto' else ''}"
                + (f" resuming: {n_frames - todo} frames already captured" if todo < n_frames else "")
            )

            done = {"n": 0}

            def progress(_idx):
                done["n"] += 1
                if done["n"] % 100 == 0 or done["n"] == todo:
                    el = time.time() - t_wall
                    log(f"{done['n']}/{todo} frames ({el:.1f}s, {done['n'] / max(el, 1e-6):.1f} fps)")

            if frames:
                # Contiguous chunks per worker page -> good tile-cache locality.
                per = -(-todo // max(1, workers))
                chunks = [frames[i : i + per] for i in range(0, todo, per)]
                pages = [(page0, pending0)] + [await open_page() for _ in chunks[1:]]
                await asyncio.gather(
                    *(
                        _render_frames(page, pending, chunk, clip, quality, capture, frame_dir, ext, progress)
                        for (page, pending), chunk in zip(pages, chunks)
                    )
                )
            await browser.close()

        seq_dir = _assemble_sequence(
            frame_dir, ext, n_frames, int(round(hold_start_s * fps)), int(round(hold_end_s * fps))
        )
        _encode_mp4(seq_dir, ext, fps, quality, out_path)
    except BaseException:
        if not capture.resume:
            shutil.rmtree(frame_dir, ignore_errors=True)
        else:
            log(f"render stopped; captured frames kept in {frame_dir} for the next run")
        raise
    shutil.rmtree(frame_dir, ignore_errors=True)

    log(f"wrote {out_path} ({time.time() - t_wall:.1f}s total)")
    return out_path


@register()
def render_animation(
    html_path: Annotated[str, Field(description="Animated map HTML from draw_animated_map.")],
    output_dir: Annotated[str | SkipJsonSchema[None], Field(description="Directory for the video.")] = None,
    out_path: Annotated[
        str, Field(description="Video file name (or path when output_dir is unset).")
    ] = "animation.mp4",
    camera: Annotated[
        CameraOptions,
        AdvancedField(
            default=FixedCamera(),
            discriminator="type_",
            title="Camera",
            description="Camera for the clip, described as in Google Maps 3D: the center it looks at, its range "
            "(distance), tilt, heading, field of view and altitude mode. Fixed view, Fly to & around, Follow a "
            "subject, or Follow the action. Blank values frame the animated data at the video's size, with the "
            "map's tilt and heading.",
        ),
    ] = FixedCamera(),
    camera_settings: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            title="Camera settings",
            description="Camera designed in the animated map's Camera panel: paste the copied settings, or give "
            "the path of the downloaded settings file. Overrides Camera.",
        ),
    ] = None,
    fps: Annotated[int, AdvancedField(default=30, gt=0, description="Output video frame rate.")] = 30,
    duration: Annotated[
        DurationConfig,
        AdvancedField(
            default=DurationConfig(),
            title="Duration",
            description="Video duration. 'auto' uses the animation's own playback length (duration_s).",
        ),
    ] = DurationConfig(),
    resolution: Annotated[
        ResolutionOptions,
        AdvancedField(
            default=PresetResolution(),
            discriminator="preset",
            title="Resolution",
            description="Output video resolution: a landscape preset (720p/1080p/4K), vertical 9:16 or square "
            "1:1 for social media, or 'custom' width/height.",
        ),
    ] = PresetResolution(),
    start_frac: Annotated[
        float, AdvancedField(default=0.0, ge=0, le=1, description="Timeline fraction to start at (0 = beginning).")
    ] = 0.0,
    end_frac: Annotated[
        float, AdvancedField(default=1.0, ge=0, le=1, description="Timeline fraction to stop at (1 = end).")
    ] = 1.0,
    hold_start_s: Annotated[
        float,
        AdvancedField(
            default=1.0, ge=0, le=30, description="Seconds to hold the opening frame before playback starts."
        ),
    ] = 1.0,
    hold_end_s: Annotated[
        float,
        AdvancedField(
            default=2.0, ge=0, le=30, description="Seconds to hold the final frame (the completed tracks) at the end."
        ),
    ] = 2.0,
    overlays: Annotated[
        VideoOverlays,
        AdvancedField(
            default=VideoOverlays(),
            title="Overlays",
            description="What the video shows over the map: the date and time, legend, north arrow, scale bar, "
            "title and marker labels.",
        ),
    ] = VideoOverlays(),
    quality: Annotated[
        VideoQuality, AdvancedField(default=VideoQuality(), description="Frame format and H.264 encoding.")
    ] = VideoQuality(),
    capture: Annotated[
        CaptureOptions, AdvancedField(default=CaptureOptions(), description="Headless-browser capture settings.")
    ] = CaptureOptions(),
    verbose: Annotated[bool, AdvancedField(default=True, description="Print progress to stderr.")] = True,
) -> str:
    """Render an animated map HTML to an MP4 video file."""
    coro = render_animation_async(
        html_path,
        output_dir=output_dir,
        out_path=out_path,
        camera=camera,
        camera_settings=camera_settings,
        fps=fps,
        duration=duration,
        resolution=resolution,
        start_frac=start_frac,
        end_frac=end_frac,
        hold_start_s=hold_start_s,
        hold_end_s=hold_end_s,
        overlays=overlays,
        quality=quality,
        capture=capture,
        verbose=verbose,
    )
    return _run_sync(coro)


def _run_sync(coro):
    """Run a coroutine to completion from sync code, also inside a running loop (Jupyter/IPython)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running: run the coroutine on a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class CameraFromSettings(BaseModel):
    """The camera designed in the animated map's Camera panel."""

    model_config = ConfigDict(json_schema_extra={"title": "From Camera panel settings"})
    type_: Annotated[Literal["settings"], Field(default="settings", title="Camera")] = "settings"
    settings: Annotated[
        str,
        Field(
            title="Camera settings",
            description="Paste the settings copied from the Camera panel, or give the path of the downloaded "
            "settings file (.json).",
        ),
    ]


@register()
def render_video_from_settings(
    html_path: Annotated[str, Field(description="Animated map HTML from draw_animated_map.")],
    camera_settings: Annotated[
        CameraOptions | CameraFromSettings,
        AdvancedField(
            default=FixedCamera(),
            discriminator="type_",
            title="Camera",
            description="Camera for the clip: Fixed view, Fly to & around, Follow a subject, Follow the action or "
            "Keyframes, or the settings of a camera designed in the map's Camera panel.",
        ),
    ] = FixedCamera(),
    fps: Annotated[int, AdvancedField(default=30, gt=0, description="Output video frame rate.")] = 30,
    duration: Annotated[
        DurationConfig,
        AdvancedField(
            default=DurationConfig(),
            title="Duration",
            description="Video duration. 'auto' uses the animation's own playback length (duration_s).",
        ),
    ] = DurationConfig(),
    overlays: Annotated[
        VideoOverlays,
        AdvancedField(
            default=VideoOverlays(),
            title="Overlays",
            description="What the video shows over the map: the date and time, legend, north arrow, scale bar, "
            "title and marker labels.",
        ),
    ] = VideoOverlays(),
    quality: Annotated[
        VideoQuality, AdvancedField(default=VideoQuality(), description="Frame format and H.264 encoding.")
    ] = VideoQuality(),
    capture: Annotated[
        CaptureOptions, AdvancedField(default=CaptureOptions(), description="Headless-browser capture settings.")
    ] = CaptureOptions(),
    output_dir: Annotated[str | SkipJsonSchema[None], Field(description="Directory for the video.")] = None,
    out_path: Annotated[
        str, Field(description="Video file name (or path when output_dir is unset).")
    ] = "animation.mp4",
    verbose: Annotated[bool, AdvancedField(default=True, description="Print progress to stderr.")] = True,
) -> str:
    """Render an animated map HTML to an MP4 video, with a camera picked here or designed in the map's
    Camera panel."""
    if isinstance(camera_settings, CameraFromSettings):
        camera = parse_camera_settings(camera_settings.settings)
    else:
        camera = camera_settings
    return _run_sync(
        render_animation_async(
            html_path,
            output_dir=output_dir,
            out_path=out_path,
            camera=camera,
            fps=fps,
            duration=duration,
            overlays=overlays,
            quality=quality,
            capture=capture,
            verbose=verbose,
        )
    )

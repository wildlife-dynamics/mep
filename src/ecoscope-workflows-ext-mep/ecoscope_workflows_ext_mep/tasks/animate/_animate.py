import asyncio
import json
import logging
import os
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
}


class PresetResolution(BaseModel):
    """Output video resolution from a common preset."""

    model_config = ConfigDict(json_schema_extra={"title": "Preset"})
    preset: Annotated[
        Literal["720p", "1080p", "4k"],
        Field(default="720p", title="Resolution", description="Common output video resolution preset."),
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


class CameraKeyframe(BaseModel):
    """One waypoint on a user-defined camera path.

    Only ``lon``/``lat`` are required. ``t`` places the keyframe on the clip
    timeline (0 = first captured frame, 1 = last); when omitted on every
    keyframe they are spaced evenly. ``zoom``/``pitch``/``bearing`` left as
    None are interpolated between the nearest keyframes that define them, or
    fall back to the whole-scene framing (zoom) and the map's pitch/bearing.
    """

    lon: Annotated[float, Field(description="Longitude of the camera look-at point.")]
    lat: Annotated[float, Field(description="Latitude of the camera look-at point.")]
    t: Annotated[
        float | None,
        Field(
            default=None,
            description="Position on the clip timeline (0 = start, 1 = end). "
            "Any monotonically increasing numbers work (they are normalized to 0–1); "
            "leave blank on every keyframe to space them evenly.",
        ),
    ] = None
    zoom: Annotated[
        float | None,
        Field(
            default=None,
            description="Zoom level at this keyframe. Leave blank to blend from neighbouring keyframes, "
            "or fit the whole scene if none set it.",
        ),
    ] = None
    pitch: Annotated[
        float | None,
        Field(
            default=None,
            description="Camera tilt in degrees at this keyframe. Leave blank to blend from neighbouring keyframes, "
            "or use the map's pitch if none set it.",
        ),
    ] = None
    bearing: Annotated[
        float | None,
        Field(
            default=None,
            description="Camera heading in degrees at this keyframe. Leave blank to blend from neighbouring "
            "keyframes, or use the map's bearing if none set it.",
        ),
    ] = None


_KF_CHANNELS = ("zoom", "pitch", "bearing")


def _load_keyframes_file(path: str) -> list[dict]:
    """Load camera keyframes from a .json (list of objects), .geojson
    (Point features; t/zoom/pitch/bearing read from properties), or .csv/.tsv
    (lon/lat columns, optional t/zoom/pitch/bearing columns) file."""
    import csv
    import json

    p = Path(remove_file_scheme(path))
    suffix = p.suffix.lower()

    def norm(d: dict) -> dict:
        out: dict = {}
        for key, aliases in (
            ("lon", ("lon", "longitude", "lng", "x")),
            ("lat", ("lat", "latitude", "y")),
            ("t", ("t", "time", "frac")),
            ("zoom", ("zoom",)),
            ("pitch", ("pitch",)),
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


def _resolve_keyframes(raw: list[dict], base_view: dict) -> list[dict]:
    """Turn sparse user keyframes into a complete, sorted camera path.

    - Missing ``t`` -> even spacing; any monotone times are normalized to 0–1.
    - Missing zoom/pitch/bearing -> interpolated between the nearest keyframes
      that define them (held flat past the ends). When no keyframe defines a
      channel, pitch/bearing come from the map and zoom stays None for the page
      to fill with the whole-scene framing at the video's size.
    - Bearings are unwrapped so in-browser lerp always rotates the short way.
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
    defaults = {
        "zoom": None,
        "pitch": base_view.get("pitch", 0),
        "bearing": base_view.get("bearing", 0),
    }
    for ch in _KF_CHANNELS:
        idxs = [i for i, k in enumerate(kf) if k.get(ch) is not None]
        if not idxs:
            for k in kf:
                k[ch] = defaults[ch]
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
    return [
        {"t": k["t"], "lon": k["lon"], "lat": k["lat"], "zoom": k["zoom"], "pitch": k["pitch"], "bearing": k["bearing"]}
        for k in kf
    ]


def keyframes_from_gdf(
    gdf,
    subject: str | int | None = None,
    n_keyframes: int = 12,
    bearing_from_travel: bool = False,
    smooth_window: int = 3,
    span: float | None = None,
) -> list[CameraKeyframe]:
    """Derive camera keyframes from a trajectory GeoDataFrame.

    Expects the same shape the animated TripsLayer is built from: one row per
    subject with a ``timestamps`` sequence and a (2D or Z) LineString whose
    vertices align with those timestamps. Zoom is left to the renderer's framing
    and pitch to the map.

    - ``subject``: row to follow — a value in any column (e.g. its name or group id),
      a positional index, ``"all"`` for the mean position of every subject, or
      None to pick the longest-running track.
    - Keyframe ``t`` is the time since the earliest timestamp divided by ``span``
      (default: the gdf's full time range) — the same timeline the animation plays
      over — so the camera is where the subject is *at that moment*. If other
      animated layers start earlier, or you trim with start_frac/end_frac, pass
      span accordingly.
    - ``bearing_from_travel``: also set each keyframe's bearing to the local
      direction of travel (the resolver unwraps them for shortest rotation).
    - ``smooth_window``: odd rolling-mean width applied to the sampled lon/lat
      to keep the camera from inheriting GPS jitter. 0/1 disables.
    """
    import numpy as np

    if n_keyframes < 2:
        raise ValueError("n_keyframes must be >= 2")

    def track(row):
        T = np.asarray(list(row.timestamps), dtype=float)
        C = np.asarray(row.geometry.coords, dtype=float)[:, :2]  # drop Z
        if len(T) != len(C):
            m = min(len(T), len(C))
            T, C = T[:m], C[:m]
        order = np.argsort(T)
        return T[order], C[order]

    tracks = [track(r) for r in gdf.itertuples(index=False)]
    if not tracks:
        raise ValueError("gdf has no rows")
    t_start = min(float(T[0]) for T, _ in tracks if len(T))
    global_span = float(span) if span else max(float(T[-1]) for T, _ in tracks if len(T)) - t_start
    if global_span <= 0:
        raise ValueError("could not determine a positive time span from the gdf")

    if isinstance(subject, str) and subject != "all":
        attrs = gdf.drop(columns=[gdf.geometry.name, "timestamps"], errors="ignore").astype(str)
        mask = (attrs == subject).any(axis=1)
        if not mask.any():
            raise ValueError(f"subject {subject!r} not found in any column")
        chosen = [tracks[i] for i in np.flatnonzero(mask.to_numpy())][:1]
    elif subject == "all":
        chosen = tracks
    elif isinstance(subject, int):
        chosen = [tracks[subject]]
    else:  # None -> longest-running track
        chosen = [max(tracks, key=lambda tc: tc[0][-1] if len(tc[0]) else -1)]

    t0 = min(float(T[0]) for T, _ in chosen)
    t1 = max(float(T[-1]) for T, _ in chosen)
    sample_times = np.linspace(t0, t1, n_keyframes)

    def at(T, C, ts):
        return np.stack([np.interp(ts, T, C[:, 0]), np.interp(ts, T, C[:, 1])], axis=1)

    pts = np.mean([at(T, C, sample_times) for T, C in chosen], axis=0)

    w = int(smooth_window)
    if w > 1:
        if w % 2 == 0:
            w += 1
        pad = w // 2
        padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(w) / w
        pts = np.stack([np.convolve(padded[:, 0], kernel, "valid"), np.convolve(padded[:, 1], kernel, "valid")], axis=1)

    bearings: list[float | None] = [None] * n_keyframes
    if bearing_from_travel:
        for i in range(n_keyframes):
            a = pts[max(0, i - 1)], pts[min(n_keyframes - 1, i + 1)]
            dx = (a[1][0] - a[0][0]) * np.cos(np.radians((a[0][1] + a[1][1]) / 2))
            dy = a[1][1] - a[0][1]
            bearings[i] = float(np.degrees(np.arctan2(dx, dy))) if (dx or dy) else None

    return [
        CameraKeyframe(
            lon=float(pts[i, 0]),
            lat=float(pts[i, 1]),
            t=float(min(1.0, max(0.0, (sample_times[i] - t_start) / global_span))),
            bearing=bearings[i],
        )
        for i in range(n_keyframes)
    ]


@register()
def derive_camera_keyframes(
    trajectory_gdf: Annotated[
        object,
        Field(description="Trajectory GeoDataFrame with per-subject LineString geometry and a 'timestamps' column."),
    ],
    subject: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Subject to follow: its value in any column (e.g. name or group id), 'all' for the group "
            "mean position, or empty for the longest-running track.",
        ),
    ] = None,
    n_keyframes: Annotated[
        int, AdvancedField(default=12, ge=2, description="Number of camera keyframes to sample along the track.")
    ] = 12,
    bearing_from_travel: Annotated[
        bool,
        AdvancedField(default=False, description="Rotate the camera to face the subject's direction of travel."),
    ] = False,
    smooth_window: Annotated[
        int,
        AdvancedField(default=3, ge=0, description="Rolling-mean window (keyframes) to smooth GPS jitter. 0 = off."),
    ] = 0,
) -> list[CameraKeyframe]:
    """Auto-generate camera keyframes for render_animation from a trajectory gdf."""
    return keyframes_from_gdf(
        trajectory_gdf,
        subject=subject,
        n_keyframes=n_keyframes,
        bearing_from_travel=bearing_from_travel,
        smooth_window=smooth_window,
    )


# Every camera frames whatever is animating -- trips, points, polygons -- at the video's
# own size, with the map's pitch and bearing; this nudges that automatic framing.
ZoomOffset = Annotated[
    float,
    AdvancedField(default=0.0, description="Zoom levels added to the automatic framing (+1 = twice as close)."),
]
SubjectFilter = Annotated[
    list[str] | SkipJsonSchema[None],
    AdvancedField(
        default=None,
        description="Only frame these subjects (their value in any column, e.g. names), zooming out as they spread "
        "apart. Leave blank for everything that animates.",
    ),
]
SubjectName = Annotated[
    str | SkipJsonSchema[None],
    Field(
        default=None,
        description="Value identifying the subject in any column (e.g. its name or group id). "
        "Leave blank to use the longest-running track.",
    ),
]
SpeedZoom = Annotated[
    float,
    AdvancedField(
        default=0.0,
        ge=0,
        le=4,
        description="Zoom levels to pull out while moving fastest, easing back in when slow or resting. 0 = off.",
    ),
]
MinHeight = Annotated[
    float | SkipJsonSchema[None],
    AdvancedField(
        default=None,
        title="Min height (m)",
        description="Keep the camera at least this many metres above what it looks at, and aim at the data's height "
        "on 3D terrain, so hills don't hide the view. Blank = off.",
    ),
]
TurnSmoothing = Annotated[
    float,
    AdvancedField(
        default=0.15,
        ge=0,
        le=1,
        description="Share of the turn towards the direction of travel made every 1/30 s: low = slow, sweeping "
        "turns, 1 = snaps.",
    ),
]
Tilt = Annotated[float, AdvancedField(default=60.0, ge=0, le=80, description="Camera tilt (0 = straight down).")]


class StaticCamera(BaseModel):
    """Holds one view that fits all the animated data for the whole clip."""

    model_config = ConfigDict(json_schema_extra={"title": "Static"})
    type_: Annotated[Literal["static"], Field(default="static", title="Camera")] = "static"
    zoom_offset: ZoomOffset = 0.0
    min_height_m: MinHeight = None


class FollowCamera(BaseModel):
    """Follows the action: frames whatever animated in the last `follow_window` of the
    timeline, re-fitting smoothly as it moves."""

    model_config = ConfigDict(json_schema_extra={"title": "Follow the action"})
    type_: Annotated[Literal["follow"], Field(default="follow", title="Camera")] = "follow"
    zoom_offset: ZoomOffset = 0.0
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
            description="Share of the gap to the action closed every 1/30 s: "
            "low = smooth but laggy, 1 = snaps to it. Independent of fps.",
        ),
    ] = 0.25
    heading_lock: Annotated[
        bool, AdvancedField(default=False, description="Rotate the camera to face the direction of travel.")
    ] = False
    subjects: SubjectFilter = None
    speed_zoom: SpeedZoom = 0.0
    fit_padding: Annotated[int, AdvancedField(default=80, ge=0, description="Padding in pixels around the action.")] = (
        80
    )
    min_height_m: MinHeight = None


class OrbitCamera(BaseModel):
    """Circles the center of all the animated data."""

    model_config = ConfigDict(json_schema_extra={"title": "Orbit"})
    type_: Annotated[Literal["orbit"], Field(default="orbit", title="Camera")] = "orbit"
    zoom_offset: ZoomOffset = 0.0
    rounds: Annotated[float, AdvancedField(default=1.0, gt=0, description="Full turns over the clip.")] = 1.0
    min_height_m: MinHeight = None


class FitCamera(BaseModel):
    """Zooms to keep everything shown so far in frame."""

    model_config = ConfigDict(json_schema_extra={"title": "Fit everything so far"})
    type_: Annotated[Literal["fit"], Field(default="fit", title="Camera")] = "fit"
    zoom_offset: ZoomOffset = 0.0
    fit_padding: Annotated[int, AdvancedField(default=80, ge=0, description="Padding in pixels around the data.")] = 80
    subjects: SubjectFilter = None
    min_height_m: MinHeight = None


class CinematicCamera(BaseModel):
    """Fly-through: opens on the whole scene, then follows the action with a slowly
    turning, tilted camera (after the Mapbox 'cinematic route' technique)."""

    model_config = ConfigDict(json_schema_extra={"title": "Cinematic fly-through"})
    type_: Annotated[Literal["cinematic"], Field(default="cinematic", title="Camera")] = "cinematic"
    zoom_offset: ZoomOffset = 0.0
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
            description="Share of the gap to the action closed every 1/30 s: "
            "low = smooth but laggy, 1 = snaps to it. Independent of fps.",
        ),
    ] = 0.25
    lead_frac: Annotated[
        float,
        AdvancedField(
            default=0.0, ge=0, le=0.5, description="Look ahead by this share of the timeline, so the camera leads."
        ),
    ] = 0.0
    bearing_mode: Annotated[
        Literal["rotate", "heading", "fixed"],
        AdvancedField(
            default="rotate",
            description="'rotate' turns at a constant rate, 'heading' faces the direction of travel, 'fixed' "
            "keeps the map's bearing.",
        ),
    ] = "rotate"
    rotate_deg: Annotated[
        float, AdvancedField(default=45.0, description="Total turn over the clip for bearing_mode='rotate'.")
    ] = 45.0
    intro_frac: Annotated[
        float,
        AdvancedField(
            default=0.12, ge=0, le=0.5, description="Share of the clip spent flying in from the whole-scene view."
        ),
    ] = 0.12
    subjects: SubjectFilter = None
    fit_padding: Annotated[int, AdvancedField(default=80, ge=0, description="Padding in pixels around the action.")] = (
        80
    )
    min_height_m: MinHeight = None


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
    """Flies through waypoints while the data animates.

    Pick a `source`: upload a waypoint file, or follow a subject. Waypoints supplied
    directly in ``keyframes`` (e.g. from derive_camera_keyframes) take priority.
    Waypoints without a zoom use the whole-scene framing (plus zoom_offset).
    """

    model_config = ConfigDict(json_schema_extra={"title": "Keyframes"})
    type_: Annotated[Literal["keyframes"], Field(default="keyframes", title="Camera")] = "keyframes"
    zoom_offset: ZoomOffset = 0.0
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
    min_height_m: MinHeight = None


class FlyAroundCamera(BaseModel):
    """Fly to a point, then circle it -- the Google Maps 3D flyCameraTo + flyCameraAround pair.

    The camera is described the Google way (center, altitude, range, tilt, heading) and
    converted to deck's view state. The fly-to uses deck's FlyToInterpolator, the same
    zoom-out-and-back-in arc; the orbit then turns at a constant rate for `rounds` turns.
    Unset values come from the data (center, framing) and the map (tilt, heading).
    """

    model_config = ConfigDict(json_schema_extra={"title": "Fly to & around"})
    type_: Annotated[Literal["fly_around"], Field(default="fly_around", title="Camera")] = "fly_around"
    zoom_offset: ZoomOffset = 0.0
    lon: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None, ge=-180, le=180, description="Center longitude. Leave blank to use the center of the data."
        ),
    ] = None
    lat: Annotated[
        float | SkipJsonSchema[None],
        Field(default=None, ge=-90, le=90, description="Center latitude. Leave blank to use the center of the data."),
    ] = None
    altitude: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Height of the center point in metres. Leave blank to use the data's mean height, so the "
            "camera circles draped tracks rather than a point under the terrain.",
        ),
    ] = None
    range_m: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=None, gt=0, description="Camera distance from the center in metres. Leave blank to fit the data."
        ),
    ] = None
    tilt: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            ge=0,
            le=80,
            description="Tilt on arrival (0 = straight down). Leave blank to use the map's pitch.",
        ),
    ] = None
    heading: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(default=None, description="Compass heading on arrival. Leave blank to use the map's bearing."),
    ] = None
    fly_frac: Annotated[
        float,
        Field(
            default=0.25,
            ge=0,
            le=1,
            description="Share of the clip spent flying in from the whole-scene view. 0 -> start at the target.",
        ),
    ] = 0.25
    rounds: Annotated[
        float, Field(default=1.0, ge=0, description="Full turns around the center after arriving. 0 -> hold.")
    ] = 1.0
    min_height_m: MinHeight = None


class ChaseCamera(BaseModel):
    """Drone shot of one subject: sits behind it at a set distance and tilt, faces where it is heading and
    lags a little behind its moves."""

    model_config = ConfigDict(json_schema_extra={"title": "Chase a subject"})
    type_: Annotated[Literal["chase"], Field(default="chase", title="Camera")] = "chase"
    zoom_offset: ZoomOffset = 0.0
    subject: SubjectName = None
    distance_m: Annotated[
        float,
        Field(default=800.0, gt=0, title="Distance (m)", description="Camera distance from the subject in metres."),
    ] = 800.0
    tilt: Tilt = 60.0
    follow_smoothing: Annotated[
        float,
        AdvancedField(
            default=0.3,
            ge=0,
            le=1,
            description="Share of the gap to the subject closed every 1/30 s: low = drifts behind, 1 = locked on.",
        ),
    ] = 0.3
    turn_smoothing: TurnSmoothing = 0.15
    speed_zoom: SpeedZoom = 0.0
    min_height_m: MinHeight = None


class TourCamera(BaseModel):
    """Visits subjects one after another: the clip is split into one stop per subject, each framing that
    subject's track so far, with a flight between stops."""

    model_config = ConfigDict(json_schema_extra={"title": "Subject tour"})
    type_: Annotated[Literal["tour"], Field(default="tour", title="Camera")] = "tour"
    zoom_offset: ZoomOffset = 0.0
    subjects: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=None,
            description="Subjects to visit, in order (their value in any column, e.g. names). Leave blank to visit "
            "every track in order of first appearance.",
        ),
    ] = None
    transition_frac: Annotated[
        float,
        AdvancedField(default=0.25, ge=0, le=0.9, description="Share of each stop spent flying in from the last one."),
    ] = 0.25
    fit_padding: Annotated[int, AdvancedField(default=80, ge=0, description="Padding in pixels around the track.")] = 80
    min_height_m: MinHeight = None


class SpotlightCamera(BaseModel):
    """Flies to each new event as it appears (rows of a time-window layer), holds on it, then goes back to
    the camera used between events."""

    model_config = ConfigDict(json_schema_extra={"title": "Event spotlight"})
    type_: Annotated[Literal["spotlight"], Field(default="spotlight", title="Camera")] = "spotlight"
    zoom_offset: ZoomOffset = 0.0
    layer: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            description="Time-window layer whose rows are the events. Leave blank to use every time-window layer.",
        ),
    ] = None
    between: Annotated[
        Literal["fit", "follow", "static"],
        Field(default="fit", description="Camera between events: fit everything so far, follow the action, or static."),
    ] = "fit"
    hold_frac: Annotated[
        float,
        AdvancedField(
            default=0.08, gt=0, le=0.5, description="Share of the clip spent on each event, flights included."
        ),
    ] = 0.08
    min_gap_frac: Annotated[
        float,
        AdvancedField(
            default=0.05,
            ge=0,
            le=1,
            description="Least share of the clip between two spotlights; events closer together are skipped.",
        ),
    ] = 0.05
    event_zoom: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(default=None, description="Zoom level on an event. Leave blank to frame about 2 km around it."),
    ] = None
    min_height_m: MinHeight = None


class PanCamera(BaseModel):
    """Slow pan and zoom across the whole scene (the 'Ken Burns' move), for short or mostly static clips."""

    model_config = ConfigDict(json_schema_extra={"title": "Pan across"})
    type_: Annotated[Literal["pan"], Field(default="pan", title="Camera")] = "pan"
    zoom_offset: ZoomOffset = 0.0
    direction: Annotated[
        Literal["west_to_east", "east_to_west", "south_to_north", "north_to_south", "diagonal"],
        Field(default="west_to_east", description="Which way the camera drifts across the scene."),
    ] = "west_to_east"
    start_zoom_offset: Annotated[
        float, AdvancedField(default=1.0, description="Zoom levels closer than the whole scene at the start.")
    ] = 1.0
    end_zoom_offset: Annotated[
        float, AdvancedField(default=0.0, description="Zoom levels closer than the whole scene at the end.")
    ] = 0.0
    easing: Annotated[
        Literal["smooth", "linear"],
        AdvancedField(default="smooth", description="'smooth' eases in and out; 'linear' moves at a constant rate."),
    ] = "smooth"
    min_height_m: MinHeight = None


class RouteCamera(BaseModel):
    """Flies along one subject's whole route at an even pace over the clip, independent of when the subject
    travelled it -- a route preview."""

    model_config = ConfigDict(json_schema_extra={"title": "Route flythrough"})
    type_: Annotated[Literal["route"], Field(default="route", title="Camera")] = "route"
    zoom_offset: ZoomOffset = 0.0
    subject: SubjectName = None
    distance_m: Annotated[
        float,
        Field(default=1500.0, gt=0, title="Distance (m)", description="Camera distance from the route in metres."),
    ] = 1500.0
    tilt: Tilt = 60.0
    look_ahead_m: Annotated[
        float,
        AdvancedField(
            default=300.0,
            ge=0,
            title="Look ahead (m)",
            description="Face the route this many metres ahead, smoothing out small wiggles.",
        ),
    ] = 300.0
    turn_smoothing: TurnSmoothing = 0.15
    easing: Annotated[
        Literal["smooth", "linear"],
        AdvancedField(default="smooth", description="'smooth' eases in and out; 'linear' moves at a constant pace."),
    ] = "smooth"
    min_height_m: MinHeight = None


SingleCameraOptions = (
    StaticCamera
    | FollowCamera
    | OrbitCamera
    | FitCamera
    | CinematicCamera
    | KeyframesCamera
    | FlyAroundCamera
    | ChaseCamera
    | TourCamera
    | SpotlightCamera
    | PanCamera
    | RouteCamera
)


class SequenceSegment(BaseModel):
    """One part of a SequenceCamera: a camera used up to a position on the clip."""

    until: Annotated[
        float,
        Field(
            default=1.0,
            gt=0,
            le=1,
            description="Clip position (0–1) where this segment ends. The last segment always runs to the end.",
        ),
    ] = 1.0
    camera: Annotated[SingleCameraOptions, Field(discriminator="type_", description="Camera for this segment.")]


class SequenceCamera(BaseModel):
    """Combines cameras over the clip: each segment uses its own camera up to its end position, blending
    smoothly into the next."""

    model_config = ConfigDict(json_schema_extra={"title": "Sequence"})
    type_: Annotated[Literal["sequence"], Field(default="sequence", title="Camera")] = "sequence"
    segments: Annotated[
        list[SequenceSegment],
        Field(min_length=1, description="Cameras in order, each with the clip position it runs until."),
    ]
    blend_frac: Annotated[
        float,
        AdvancedField(default=0.04, ge=0, le=0.5, description="Share of the clip spent blending between segments."),
    ] = 0.04
    min_height_m: MinHeight = None


CameraOptions = SingleCameraOptions | SequenceCamera
CameraConfig = Annotated[CameraOptions, Field(discriminator="type_")]
_CAMERA_ADAPTER: TypeAdapter = TypeAdapter(CameraConfig)

# Fields the animated map's Camera panel edits with its own controls rather than a form input.
_PANEL_CUSTOM_FIELDS = ("type_", "keyframes", "source", "segments")


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


# --- JS injected into each page: reads the scene data and builds the camera ----
# Parses window.jsonInput (the pydeck spec, a real global). `path(times, opts)`
# returns the full per-frame viewState array in a single call.
_CAM_HELPER = r"""
window.__cam = (function () {
  // ---------------------------------------------------------------------------
  // Scene: every animated layer, whatever its type. Trips contribute per-vertex
  // times; time-window layers (points, paths, polygons, ...) one time per row.
  // Layer specs come from draw_animated_map via window.__tripsAnim.specs.
  // ---------------------------------------------------------------------------
  var SCENE = null;
  var MAX_ZOOM = 16;          // never frame tighter than this
  var MIN_BOX_DEG = 0.02;     // frame at least ~2 km, so a lone point is not a close-up
  var EARTH_M = 40075016.686; // equator length; deck's world is 512 * 2^zoom px wide
  var M_PER_DEG = 111320;

  function rowTime(d) {
    if (d.__t !== undefined) return d.__t;
    return (d.properties && d.properties.__t !== undefined) ? d.properties.__t : NaN;
  }
  function flatten(c, out) {   // any GeoJSON coordinates -> [[lon, lat, z?], ...]
    if (!c || !c.length) return;
    if (typeof c[0] === 'number') { out.push(c); return; }
    for (var i = 0; i < c.length; i++) flatten(c[i], out);
  }
  function values(d) {         // every scalar property of a row, for subject matching
    var v = [], src = [d, d.properties || {}];
    for (var s = 0; s < src.length; s++) for (var k in src[s]) {
      var x = src[s][k];
      if (typeof x === 'string' || typeof x === 'number') v.push(String(x));
    }
    return v;
  }
  function matches(d, keys) {  // keys: null (everything) or subject values
    if (!keys) return true;
    var v = values(d);
    for (var i = 0; i < keys.length; i++) if (v.indexOf(keys[i]) >= 0) return true;
    return false;
  }
  function subjectKeys(o) {
    var s = o.subjects;
    return (s && s.length) ? s.map(String) : null;
  }
  function scene() {
    if (SCENE) return SCENE;
    var layers = (window.jsonInput && window.jsonInput.layers) || [];
    var specs = (window.__tripsAnim && window.__tripsAnim.specs) || [];
    if (!specs.length) {       // older pages: animate the first TripsLayer only
      var tl = layers.find(function (L) { return L && (L['@@type'] === 'TripsLayer' || ('currentTime' in L)); });
      if (tl) specs = [{ id: tl.id, kind: 'trips' }];
    }
    var tracks = [], rows = [], all = [], end = 0;
    specs.forEach(function (s) {
      var L = layers.find(function (l) { return l.id === s.id; });
      var data = (L && Array.isArray(L.data)) ? L.data : [];
      data.forEach(function (d) {
        if (s.kind === 'trips') {
          var C = d.geometry && d.geometry.coordinates, T = d.timestamps;
          if (!C || !T || !C.length) return;
          tracks.push({ d: d, C: C, T: T, id: s.id });
          for (var k = 0; k < C.length; k++) all.push(C[k]);
          end = Math.max(end, T[T.length - 1]);
        } else {
          var t = rowTime(d), P = [];
          flatten(d.geometry && d.geometry.coordinates, P);
          if (!P.length || !isFinite(t)) return;
          rows.push({ d: d, t: t, P: P, w: s.window, id: s.id });
          for (var j = 0; j < P.length; j++) all.push(P[j]);
          end = Math.max(end, t);
        }
      });
    });
    var b = new Box(); all.forEach(function (p) { b.add(p); });
    var sz = 0, hasZ = false;
    all.forEach(function (p) { sz += p[2] || 0; if (p[2]) hasZ = true; });
    var withZ = all.filter(function (p) { return p.length > 2 && isFinite(p[2]); });
    var stride = Math.max(1, Math.ceil(withZ.length / 4000)), pts = [];   // sample for ground heights
    for (var q = 0; q < withZ.length; q += stride) pts.push(withZ[q]);
    SCENE = { tracks: tracks, rows: rows, box: b, meanZ: all.length ? sz / all.length : 0, end: end,
              hasZ: hasZ, pts: pts };
    return SCENE;
  }

  function Box() { this.w = 180; this.s = 90; this.e = -180; this.n = -90; this.empty = true; }
  Box.prototype.add = function (p) {
    this.w = Math.min(this.w, p[0]); this.e = Math.max(this.e, p[0]);
    this.s = Math.min(this.s, p[1]); this.n = Math.max(this.n, p[1]); this.empty = false;
  };
  Box.prototype.center = function () { return { lon: (this.w + this.e) / 2, lat: (this.s + this.n) / 2 }; };

  function lowerBound(T, t) {
    var lo = 0, hi = T.length;
    while (lo < hi) { var m = (lo + hi) >> 1; if (T[m] < t) lo = m + 1; else hi = m; }
    return lo;
  }
  function posAt(tr, t) {      // interpolated [lon, lat, z] of a track at time t (clamped)
    var C = tr.C, T = tr.T, last = T.length - 1;
    if (t <= T[0]) return C[0];
    if (t >= T[last]) return C[last];
    var j = Math.max(0, lowerBound(T, t) - 1), f = (T[j + 1] > T[j]) ? (t - T[j]) / (T[j + 1] - T[j]) : 0;
    var a = C[j], b = C[j + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, (a[2] || 0) + ((b[2] || 0) - (a[2] || 0)) * f];
  }
  function groundDist(a, b) {  // metres between two [lon, lat] points (small-distance approximation)
    var lat = (a[1] + b[1]) * 0.5 * Math.PI / 180;
    return Math.hypot((b[0] - a[0]) * Math.cos(lat), b[1] - a[1]) * M_PER_DEG;
  }
  function headingLL(a, b) {   // compass heading from a to b, or null when they coincide
    if (!a || !b || (a[0] === b[0] && a[1] === b[1])) return null;
    var lat = (a[1] + b[1]) * 0.5 * Math.PI / 180;
    return Math.atan2((b[0] - a[0]) * Math.cos(lat), b[1] - a[1]) * 180 / Math.PI;
  }

  // What's on screen around t: track vertices from [t - w, t] + current positions, and the
  // rows each time-window layer is showing (its own window, or w for cumulative layers).
  function recentBox(t, w, keys) {
    var S = scene(), b = new Box();
    S.tracks.forEach(function (tr) {
      if (!matches(tr.d, keys)) return;
      if (t < tr.T[0] || t - w > tr.T[tr.T.length - 1]) return;     // not started / long finished
      for (var k = lowerBound(tr.T, t - w); k < tr.T.length && tr.T[k] <= t; k++) b.add(tr.C[k]);
      b.add(posAt(tr, t));
    });
    S.rows.forEach(function (r) {
      if (!matches(r.d, keys)) return;
      var from = t - (r.w == null ? w : Math.max(r.w, w));
      if (r.t >= from && r.t <= t) r.P.forEach(function (p) { b.add(p); });
    });
    return b;
  }
  // Everything shown up to t (tracks' traversed paths, rows that have appeared).
  function seenBox(t, keys) {
    var S = scene(), b = new Box();
    S.tracks.forEach(function (tr) {
      if (!matches(tr.d, keys)) return;
      for (var k = 0; k < tr.T.length && tr.T[k] <= t; k++) b.add(tr.C[k]);
    });
    S.rows.forEach(function (r) { if (r.t <= t && matches(r.d, keys)) r.P.forEach(function (p) { b.add(p); }); });
    return b;
  }

  // Fit a box to the video frame; boxes under MIN_BOX_DEG are widened around their center.
  function fitBox(b, o) {
    if (!b || b.empty) return null;
    var c = b.center(), VP = (window.deck || {}).WebMercatorViewport, z = MAX_ZOOM;
    var hw = Math.max(b.e - b.w, MIN_BOX_DEG) / 2, hh = Math.max(b.n - b.s, MIN_BOX_DEG) / 2;
    if (VP) {
      try {
        var f = new VP({ width: o.width, height: o.height })
          .fitBounds([[c.lon - hw, c.lat - hh], [c.lon + hw, c.lat + hh]],
                     { padding: o.fit_padding == null ? 80 : o.fit_padding });
        c = { lon: f.longitude, lat: f.latitude }; z = Math.min(f.zoom, MAX_ZOOM);
      } catch (e) {}
    }
    return { longitude: c.lon, latitude: c.lat, zoom: z + (o.zoom_offset || 0) };
  }

  function span() { return scene().end; }
  function initialView() { return (window.jsonInput && window.jsonInput.initialViewState) || null; }

  function shortestAngle(a, b) { var d = ((b - a + 180) % 360) - 180; return d <= -180 ? d + 360 : d; }
  function travelHeading(a, b) {
    if (!a || !b) return null;
    return headingLL([a.longitude, a.latitude], [b.longitude, b.latitude]);
  }
  function clamp01(u) { return Math.max(0, Math.min(1, u)); }
  function ease(u) { u = clamp01(u); return u * u * (3 - 2 * u); }
  // A "share of the gap closed every 1/30 s" setting, rescaled to the frame rate.
  function rate(share, fallback, fps) {
    var s = clamp01(share == null ? fallback : share);
    return s > 0 ? 1 - Math.pow(1 - s, 30 / (fps || 30)) : 1;
  }
  // Google's `range` (camera-to-center metres) -> deck zoom. deck's MapView keeps the
  // camera 1.5 viewport-heights from the target, and the world is 512 * 2^zoom px wide.
  function zoomForRange(rangeM, lat, height) {
    return Math.log2(1.5 * height * EARTH_M * Math.cos(lat * Math.PI / 180) / (512 * rangeM));
  }
  function rangeForZoom(zoom, lat, height) {
    return 1.5 * height * EARTH_M * Math.cos(lat * Math.PI / 180) / (512 * Math.pow(2, zoom));
  }
  function copies(v, n) { var out = []; for (var i = 0; i < n; i++) out.push(Object.assign({}, v)); return out; }

  // Blend two views; bearings turn the short way and target heights blend too.
  function lerpView(a, b, e) {
    var v = { longitude: a.longitude + (b.longitude - a.longitude) * e,
              latitude:  a.latitude  + (b.latitude  - a.latitude ) * e,
              zoom:      a.zoom      + (b.zoom      - a.zoom     ) * e,
              pitch:   (a.pitch || 0) + ((b.pitch || 0) - (a.pitch || 0)) * e,
              bearing: (a.bearing || 0) + shortestAngle(a.bearing || 0, b.bearing || 0) * e };
    if (a.position || b.position) {
      var za = a.position ? a.position[2] : 0, zb = b.position ? b.position[2] : 0;
      v.position = [0, 0, za + (zb - za) * e];
    }
    return v;
  }
  // Same, but along deck's FlyToInterpolator arc (zoom out, travel, zoom in) when available.
  function flyView(fly, a, b, e, o) {
    var v = lerpView(a, b, e);
    if (fly && e > 0 && e < 1) {
      try {
        var dims = { width: o.width, height: o.height };
        var f = fly.interpolateProps(Object.assign({}, dims, a), Object.assign({}, dims, b), e);
        v.longitude = f.longitude; v.latitude = f.latitude; v.zoom = f.zoom;
      } catch (err) {}
    }
    return v;
  }
  function flyer() { var FTI = (window.deck || {}).FlyToInterpolator; return FTI ? new FTI({ curve: 1.414 }) : null; }

  // ---------------------------------------------------------------------------
  // Subjects
  // ---------------------------------------------------------------------------
  // One subject as a track ({T, C, d}): a matching trips track, else the time-ordered
  // centres of matching rows of point/polygon layers. No subject -> the longest-running track.
  function subjectTrack(subject) {
    var S = scene(), key = (subject == null || subject === '') ? null : String(subject);
    var tr = null;
    if (key == null) {
      S.tracks.forEach(function (x) {
        if (!tr || x.T[x.T.length - 1] - x.T[0] > tr.T[tr.T.length - 1] - tr.T[0]) tr = x;
      });
    } else {
      tr = S.tracks.find(function (x) { return values(x.d).indexOf(key) >= 0; }) || null;
    }
    if (tr) return tr;
    var rows = key == null ? [] : S.rows.filter(function (r) { return values(r.d).indexOf(key) >= 0; });
    if (!rows.length) return null;
    rows.sort(function (a, b) { return a.t - b.t; });
    return {
      d: rows[0].d,
      T: rows.map(function (r) { return r.t; }),
      C: rows.map(function (r) {
        var b = new Box(), z = 0; r.P.forEach(function (p) { b.add(p); z += p[2] || 0; });
        var c = b.center(); return [c.lon, c.lat, z / r.P.length];
      }),
    };
  }
  // Height of the data nearest a point (draped tracks carry the terrain height).
  function groundZ(lon, lat) {
    var S = scene(), best = Infinity, z = S.meanZ, k = Math.cos(lat * Math.PI / 180);
    S.pts.forEach(function (p) {
      var d = Math.pow((p[0] - lon) * k, 2) + Math.pow(p[1] - lat, 2);
      if (d < best) { best = d; z = p[2] || 0; }
    });
    return z;
  }

  // ---------------------------------------------------------------------------
  // Keyframes
  // ---------------------------------------------------------------------------
  function catmullRom(p0, p1, p2, p3, u) {
    return 0.5 * ((2 * p1) + (-p0 + p2) * u
           + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
           + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u);
  }
  // Keyframes arrive resolved from Python: sorted, t in [0,1], bearings unwrapped;
  // zoom may be null there and is filled with the whole-scene framing.
  function keyframeView(K, prog, easing) {
    var hi = K.length - 1, s = 0;
    while (s + 1 < hi && K[s + 1].t <= prog) s++;
    var A = K[s], B = K[Math.min(s + 1, hi)];
    var u = (B.t > A.t) ? (prog - A.t) / (B.t - A.t) : 1;
    u = clamp01(u);
    var eased = (easing === 'linear' || easing === 'spline') ? u : u * u * (3 - 2 * u);
    var lon, lat;
    if (easing === 'spline') {
      var P0 = K[Math.max(0, s - 1)], P3 = K[Math.min(hi, s + 2)];
      lon = catmullRom(P0.lon, A.lon, B.lon, P3.lon, u);
      lat = catmullRom(P0.lat, A.lat, B.lat, P3.lat, u);
    } else {
      lon = A.lon + (B.lon - A.lon) * eased;
      lat = A.lat + (B.lat - A.lat) * eased;
    }
    return { longitude: lon, latitude: lat,
             zoom:    A.zoom    + (B.zoom    - A.zoom   ) * eased,
             pitch:   A.pitch   + (B.pitch   - A.pitch  ) * eased,
             bearing: A.bearing + (B.bearing - A.bearing) * eased };
  }
  function autoKeyframes(o, base) {
    var tr = subjectTrack(o.subject);
    if (!tr || tr.T.length < 2) return [];
    var n = o.auto_keyframe_count || 12, END = span() || 1, b = new Box();
    tr.C.forEach(function (p) { b.add(p); });
    var zoom = (fitBox(b, o) || base).zoom, out = [], T0 = tr.T[0], T1 = tr.T[tr.T.length - 1];
    for (var i = 0; i < n; i++) {
      var tt = T0 + (T1 - T0) * (i / (n - 1)), p = posAt(tr, tt);
      out.push({ t: Math.min(1, tt / END), lon: p[0], lat: p[1], zoom: zoom,
                 pitch: base.pitch, bearing: base.bearing });
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Fly to & around (Google Maps 3D flyCameraTo + flyCameraAround)
  // ---------------------------------------------------------------------------
  function flyAroundSetup(o, base) {
    var S = scene(), c = S.box.empty ? { lon: base.longitude, lat: base.latitude } : S.box.center();
    var lon = (o.lon != null) ? o.lon : c.lon, lat = (o.lat != null) ? o.lat : c.lat;
    var zoom = (o.range_m != null) ? zoomForRange(o.range_m, lat, o.height) + (o.zoom_offset || 0) : base.zoom;
    var end = { longitude: lon, latitude: lat, zoom: zoom,
                pitch: (o.tilt != null) ? o.tilt : base.pitch,
                bearing: base.bearing + shortestAngle(base.bearing, (o.heading != null) ? o.heading : base.bearing) };
    return { start: base, end: end, alt: (o.altitude != null) ? o.altitude : S.meanZ, fly: flyer(),
             frac: o.fly_frac == null ? 0.25 : o.fly_frac, rounds: o.rounds == null ? 1 : o.rounds,
             width: o.width, height: o.height };
  }
  function flyAroundView(A, prog) {
    if (A.frac > 0 && prog < A.frac) {                     // flyCameraTo
      var s = prog / A.frac, e = s * s * (3 - 2 * s), v = null;
      if (A.fly) {
        try {
          var dims = { width: A.width, height: A.height };
          v = A.fly.interpolateProps(Object.assign({}, dims, A.start), Object.assign({}, dims, A.end), e);
        } catch (err) { v = null; }
      }
      if (!v) v = { longitude: A.start.longitude + (A.end.longitude - A.start.longitude) * e,
                    latitude:  A.start.latitude  + (A.end.latitude  - A.start.latitude ) * e,
                    zoom:      A.start.zoom      + (A.end.zoom      - A.start.zoom     ) * e };
      return { longitude: v.longitude, latitude: v.latitude, zoom: v.zoom,
               pitch:   A.start.pitch   + (A.end.pitch   - A.start.pitch  ) * e,
               bearing: A.start.bearing + (A.end.bearing - A.start.bearing) * e,
               position: [0, 0, A.alt * e] };
    }
    var u = (A.frac < 1) ? (prog - A.frac) / (1 - A.frac) : 1;   // flyCameraAround
    return { longitude: A.end.longitude, latitude: A.end.latitude, zoom: A.end.zoom, pitch: A.end.pitch,
             bearing: (A.end.bearing + 360 * A.rounds * u) % 360, position: [0, 0, A.alt] };
  }

  // ---------------------------------------------------------------------------
  // Chase: behind one subject, facing where it is heading
  // ---------------------------------------------------------------------------
  function chasePath(times, o, base) {
    var tr = subjectTrack(o.subject);
    if (!tr) return copies(base, times.length);
    var dt = Math.max((span() || 1) * 0.005, 1e-6);
    var dist = o.distance_m || 800, tilt = o.tilt == null ? 60 : o.tilt;
    var kPos = rate(o.follow_smoothing, 0.3, o.fps), kTurn = rate(o.turn_smoothing, 0.15, o.fps);
    var cur = null, brg = null;
    return times.map(function (t) {
      var p = posAt(tr, t), h = headingLL(posAt(tr, t - dt), posAt(tr, t + dt));
      if (brg == null) brg = (h == null) ? base.bearing : h;
      else if (h != null) brg += shortestAngle(brg, h) * kTurn;
      cur = cur ? [cur[0] + (p[0] - cur[0]) * kPos, cur[1] + (p[1] - cur[1]) * kPos,
                   cur[2] + ((p[2] || 0) - cur[2]) * kPos]
                : [p[0], p[1], p[2] || 0];
      return { longitude: cur[0], latitude: cur[1], zoom: zoomForRange(dist, cur[1], o.height) + (o.zoom_offset || 0),
               pitch: tilt, bearing: brg, position: [0, 0, cur[2]] };
    });
  }

  // ---------------------------------------------------------------------------
  // Route flythrough: along a subject's whole route at an even pace
  // ---------------------------------------------------------------------------
  function routePath(times, o, base) {
    var tr = subjectTrack(o.subject);
    if (!tr || tr.C.length < 2) return copies(base, times.length);
    var C = tr.C, cum = [0];
    for (var k = 1; k < C.length; k++) cum.push(cum[k - 1] + groundDist(C[k - 1], C[k]));
    var total = cum[cum.length - 1];
    function along(d) {
      var j = Math.max(1, Math.min(C.length - 1, lowerBound(cum, d)));
      var seg = cum[j] - cum[j - 1], f = seg > 0 ? (d - cum[j - 1]) / seg : 0;
      f = clamp01(f);
      var a = C[j - 1], b = C[j];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, (a[2] || 0) + ((b[2] || 0) - (a[2] || 0)) * f];
    }
    var dist = o.distance_m || 1500, tilt = o.tilt == null ? 60 : o.tilt;
    var look = o.look_ahead_m == null ? 300 : o.look_ahead_m, kTurn = rate(o.turn_smoothing, 0.15, o.fps);
    var n = times.length, brg = null;
    return times.map(function (t, i) {
      var prog = n > 1 ? i / (n - 1) : 1;
      var d = total * (o.easing === 'linear' ? prog : ease(prog));
      var p = along(d), h = headingLL(along(Math.max(0, d - look)), along(Math.min(total, d + Math.max(look, 1))));
      if (brg == null) brg = (h == null) ? base.bearing : h;
      else if (h != null) brg += shortestAngle(brg, h) * kTurn;
      return { longitude: p[0], latitude: p[1], zoom: zoomForRange(dist, p[1], o.height) + (o.zoom_offset || 0),
               pitch: tilt, bearing: brg, position: [0, 0, p[2]] };
    });
  }

  // ---------------------------------------------------------------------------
  // Subject tour: one stop per subject, flying between stops
  // ---------------------------------------------------------------------------
  function tourPath(times, o, base) {
    var S = scene(), keys = subjectKeys(o), list = [];
    if (keys) keys.forEach(function (k) { var tr = subjectTrack(k); if (tr) list.push(tr); });
    else list = S.tracks.slice().sort(function (a, b) { return a.T[0] - b.T[0]; });
    if (!list.length) return copies(base, times.length);
    var N = list.length, n = times.length, tf = clamp01(o.transition_frac == null ? 0.25 : o.transition_frac);
    var fly = flyer();
    function stop(tr, t) {      // the subject's track so far, or all of it before it starts
      var b = new Box();
      for (var k = 0; k < tr.T.length && tr.T[k] <= t; k++) b.add(tr.C[k]);
      if (b.empty) tr.C.forEach(function (p) { b.add(p); }); else b.add(posAt(tr, t));
      var f = fitBox(b, o);
      return { longitude: f.longitude, latitude: f.latitude, zoom: f.zoom, pitch: base.pitch, bearing: base.bearing };
    }
    return times.map(function (t, i) {
      var prog = n > 1 ? i / (n - 1) : 1, s = Math.min(N - 1, Math.floor(prog * N)), u = prog * N - s;
      var v = stop(list[s], t);
      if (tf > 0 && u < tf) v = flyView(fly, s > 0 ? stop(list[s - 1], t) : base, v, ease(u / tf), o);
      return v;
    });
  }

  // ---------------------------------------------------------------------------
  // Event spotlight: fly to each new event, hold, return to the camera between events
  // ---------------------------------------------------------------------------
  function spotlightPath(times, o, base) {
    var S = scene(), END = span() || 1, n = times.length;
    var between = path(times, Object.assign({}, o, { preset: o.between || 'fit', min_height_m: null, speed_zoom: 0 }));
    var hold = (o.hold_frac == null ? 0.08 : o.hold_frac) * END;
    var gap = Math.max(hold, (o.min_gap_frac == null ? 0.05 : o.min_gap_frac) * END);
    var events = S.rows.filter(function (r) { return !o.layer || r.id === o.layer; })
                       .sort(function (a, b) { return a.t - b.t; });
    var picks = [], last = -Infinity;
    events.forEach(function (r) {
      if (r.t - last < gap) return;
      last = r.t;
      var b = new Box(); r.P.forEach(function (p) { b.add(p); });
      var f = fitBox(b, o), c = b.center();
      var zoom = o.event_zoom != null ? o.event_zoom + (o.zoom_offset || 0) : (f ? f.zoom : base.zoom);
      picks.push({ from: r.t - 0.3 * hold, to: r.t + 0.7 * hold,
                   view: { longitude: f ? f.longitude : c.lon, latitude: f ? f.latitude : c.lat, zoom: zoom,
                           pitch: base.pitch, bearing: base.bearing } });
    });
    var fly = flyer();
    return times.map(function (t, i) {
      var v = between[i], p = picks.find(function (x) { return t >= x.from && t <= x.to; });
      if (!p) return v;
      var u = (t - p.from) / (p.to - p.from), w = u < 0.3 ? ease(u / 0.3) : (u > 0.7 ? ease((1 - u) / 0.3) : 1);
      return flyView(fly, v, p.view, w, o);
    });
  }

  // ---------------------------------------------------------------------------
  // Pan across: slow drift and zoom over the whole scene
  // ---------------------------------------------------------------------------
  function panPath(times, o, base) {
    var b = scene().box, n = times.length;
    if (b.empty) return copies(base, n);
    var c = b.center(), dx = (b.e - b.w) / 4, dy = (b.n - b.s) / 4;
    var ends = {
      west_to_east:   [[c.lon - dx, c.lat], [c.lon + dx, c.lat]],
      east_to_west:   [[c.lon + dx, c.lat], [c.lon - dx, c.lat]],
      south_to_north: [[c.lon, c.lat - dy], [c.lon, c.lat + dy]],
      north_to_south: [[c.lon, c.lat + dy], [c.lon, c.lat - dy]],
      diagonal:       [[c.lon - dx, c.lat - dy], [c.lon + dx, c.lat + dy]],
    }[o.direction || 'west_to_east'] || [[c.lon, c.lat], [c.lon, c.lat]];
    var z0 = base.zoom + (o.start_zoom_offset == null ? 1 : o.start_zoom_offset);
    var z1 = base.zoom + (o.end_zoom_offset == null ? 0 : o.end_zoom_offset);
    return times.map(function (t, i) {
      var prog = n > 1 ? i / (n - 1) : 1, e = o.easing === 'linear' ? prog : ease(prog);
      return { longitude: ends[0][0] + (ends[1][0] - ends[0][0]) * e,
               latitude:  ends[0][1] + (ends[1][1] - ends[0][1]) * e,
               zoom: z0 + (z1 - z0) * e, pitch: base.pitch, bearing: base.bearing };
    });
  }

  // ---------------------------------------------------------------------------
  // Sequence: each segment's camera over its part of the clip, blended at the joins
  // ---------------------------------------------------------------------------
  function sequencePath(times, o, base) {
    var segs = o.segments || [], n = times.length;
    if (!segs.length) return copies(base, n);
    var blend = Math.max(0, o.blend_frac == null ? 0.04 : o.blend_frac), prev = 0, bounds = [];
    segs.forEach(function (s, i) {
      var end = (i === segs.length - 1) ? 1 : Math.max(prev, Math.min(1, s.until == null ? 1 : s.until));
      bounds.push([prev, end]);
      prev = end;
    });
    var progs = times.map(function (t, j) { return n > 1 ? j / (n - 1) : 1; });
    // Each segment runs over its own range (plus half a blend each side), so progress-based
    // cameras (orbit, keyframes, fly-around, ...) play out within their segment.
    var parts = segs.map(function (s, i) {
      var a = Math.max(0, bounds[i][0] - blend / 2), b = Math.min(1, bounds[i][1] + blend / 2), idx = {}, sub = [];
      progs.forEach(function (p, j) { if (p >= a && p <= b) { idx[j] = sub.length; sub.push(times[j]); } });
      var so = Object.assign({ width: o.width, height: o.height, fps: o.fps }, s.camera);
      return { idx: idx, views: sub.length ? path(sub, so) : [] };
    });
    function viewOf(i, j) { var k = parts[i].idx[j]; return k == null ? null : parts[i].views[k]; }
    return progs.map(function (p, j) {
      var i = 0;
      while (i < segs.length - 1 && p >= bounds[i][1]) i++;
      var v = viewOf(i, j) || base;
      if (blend > 0 && i > 0 && p < bounds[i][0] + blend / 2) {
        var a = viewOf(i - 1, j);
        if (a) v = lerpView(a, v, ease((p - (bounds[i][0] - blend / 2)) / blend));
      }
      if (blend > 0 && i < segs.length - 1 && p > bounds[i][1] - blend / 2) {
        var b = viewOf(i + 1, j);
        if (b) v = lerpView(v, b, ease((p - (bounds[i][1] - blend / 2)) / blend));
      }
      return v;
    });
  }

  // ---------------------------------------------------------------------------
  // Shared finishing steps
  // ---------------------------------------------------------------------------
  // Pull the zoom out with the camera's ground speed (normalised to its 95th percentile).
  function speedZoom(out, times, amount, k) {
    var n = out.length;
    if (!amount || n < 3) return;
    var sp = out.map(function (v, i) {
      var a = out[Math.max(0, i - 1)], b = out[Math.min(n - 1, i + 1)];
      var dt = times[Math.min(n - 1, i + 1)] - times[Math.max(0, i - 1)];
      return dt > 0 ? groundDist([a.longitude, a.latitude], [b.longitude, b.latitude]) / dt : 0;
    });
    var ref = sp.slice().sort(function (a, b) { return a - b; })[Math.floor(0.95 * (n - 1))];
    if (!(ref > 0)) return;
    var z = 0;
    out.forEach(function (v, i) {
      z += (amount * Math.min(1, sp[i] / ref) - z) * k;
      out[i] = Object.assign({}, v, { zoom: v.zoom - z });
    });
  }
  // Keep the camera at least min_height_m above its target, aimed at the data's height.
  function keepAbove(out, o) {
    var minH = o.min_height_m, S = scene();
    if (!(minH > 0)) return;
    out.forEach(function (v, i) {
      v = Object.assign({}, v);
      if (S.hasZ && !v.position) v.position = [0, 0, groundZ(v.longitude, v.latitude)];
      var D = rangeForZoom(v.zoom, v.latitude, o.height);       // eye-to-target metres
      if (D < minH) { v.zoom = zoomForRange(minH, v.latitude, o.height); D = minH; }
      var maxPitch = Math.acos(Math.min(1, minH / D)) * 180 / Math.PI;
      if ((v.pitch || 0) > maxPitch) v.pitch = maxPitch;
      out[i] = v;
    });
  }

  // ---------------------------------------------------------------------------
  // path(times, opts) -> one viewState per frame
  // ---------------------------------------------------------------------------
  function path(times, o) {
    o = o || {};
    var preset = o.preset, init = initialView() || {}, S = scene();
    var pitch = init.pitch || 0, bearing = init.bearing || 0;          // from the map
    var whole = fitBox(S.box, o) || { longitude: init.longitude, latitude: init.latitude, zoom: init.zoom || 8 };
    var base = {
      longitude: whole.longitude, latitude: whole.latitude, zoom: whole.zoom, pitch: pitch, bearing: bearing
    };
    var END = span() || 1, win = (o.follow_window == null ? 0.1 : o.follow_window) * END;
    // follow_smoothing is per 1/30 s; rescale so the camera keeps up equally at any fps.
    var k = rate(o.follow_smoothing, 0.25, o.fps), keys = subjectKeys(o);
    var out;

    if (preset === 'chase') out = chasePath(times, o, base);
    else if (preset === 'route') out = routePath(times, o, base);
    else if (preset === 'tour') out = tourPath(times, o, base);
    else if (preset === 'spotlight') out = spotlightPath(times, o, base);
    else if (preset === 'pan') out = panPath(times, o, base);
    else if (preset === 'sequence') out = sequencePath(times, o, base);
    else {
      var KF = null, FA = null;
      if (preset === 'keyframes') {
        KF = (o.keyframes && o.keyframes.length >= 2) ? o.keyframes : autoKeyframes(o, base);
        KF.forEach(function (q) { if (q.zoom == null) q.zoom = base.zoom; });
      }
      if (preset === 'fly_around') FA = flyAroundSetup(o, base);

      var cur = null, prevTarget = null, cb = bearing, n = times.length;
      out = [];
      var follow = function (t) {                          // smoothed framing of recent action
        var f = fitBox(recentBox(t, win, keys), o);
        if (!f) return cur || base;
        cur = cur ? { longitude: cur.longitude + (f.longitude - cur.longitude) * k,
                      latitude:  cur.latitude  + (f.latitude  - cur.latitude ) * k,
                      zoom:      cur.zoom      + (f.zoom      - cur.zoom     ) * k } : f;
        return cur;
      };
      var faceTravel = function (target) {                 // smoothed heading of the camera's motion
        var h = travelHeading(prevTarget, target);
        prevTarget = target;
        if (h != null) cb += shortestAngle(cb, h) * Math.max(k, 0.12);
        return cb;
      };

      for (var i = 0; i < n; i++) {
        var t = times[i], prog = n > 1 ? i / (n - 1) : 1, vs;
        if (preset === 'follow') {
          var f = follow(t);
          vs = { longitude: f.longitude, latitude: f.latitude, zoom: f.zoom, pitch: pitch,
                 bearing: o.heading_lock ? faceTravel(f) : bearing };
        } else if (preset === 'orbit') {
          vs = Object.assign({}, base, { bearing: (bearing + 360 * (o.rounds == null ? 1 : o.rounds) * prog) % 360 });
        } else if (preset === 'fit') {
          var fb = fitBox(seenBox(t, keys), o) || base;
          vs = { longitude: fb.longitude, latitude: fb.latitude, zoom: fb.zoom, pitch: pitch, bearing: bearing };
        } else if (preset === 'keyframes') {
          vs = (KF.length >= 2) ? keyframeView(KF, prog, o.keyframe_easing || 'smooth') : Object.assign({}, base);
        } else if (preset === 'cinematic') {
          var lead = (o.lead_frac || 0) * END;
          var ft = follow(Math.min(END, t + lead));
          var mode = o.bearing_mode || 'rotate', brg;
          if (mode === 'heading') brg = faceTravel(ft);
          else if (mode === 'fixed') brg = bearing;
          else brg = bearing + (o.rotate_deg == null ? 45 : o.rotate_deg) * prog;
          vs = { longitude: ft.longitude, latitude: ft.latitude, zoom: ft.zoom, pitch: pitch, bearing: brg };
          var intro = (o.intro_frac == null ? 0.12 : o.intro_frac);
          if (intro > 0 && prog < intro) {                 // fly in from the whole-scene view
            var s = prog / intro, e = s * s * (3 - 2 * s);
            vs = { longitude: base.longitude + (vs.longitude - base.longitude) * e,
                   latitude:  base.latitude  + (vs.latitude  - base.latitude ) * e,
                   zoom:      base.zoom      + (vs.zoom      - base.zoom     ) * e,
                   pitch: pitch, bearing: vs.bearing };
          }
        } else if (preset === 'fly_around') {
          vs = flyAroundView(FA, prog);
        } else {
          vs = Object.assign({}, base);                    // static / unknown
        }
        out.push(vs);
      }
    }

    if (o.speed_zoom) speedZoom(out, times, o.speed_zoom, rate(0.08, 0.08, o.fps));
    keepAbove(out, o);
    return out;
  }

  return { span: span, initialView: initialView, path: path };
})();
"""


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


async def _prepare_page(browser, html_uri, *, width, height, device_scale_factor, head_ready_timeout_ms):
    """Open a page, load the scene, take control of the autoplay loop, inject helper.
    Returns (page, pending_counter_dict)."""
    page = await browser.new_page(
        viewport={"width": width, "height": height},
        device_scale_factor=device_scale_factor,
    )
    # Count in-flight tile/image requests so we can wait for the basemap to paint.
    pending = {"n": 0}

    def _on_req(req):
        if req.resource_type == "image":
            pending["n"] += 1

    def _on_done(req):
        if req.resource_type == "image":
            pending["n"] = max(0, pending["n"] - 1)

    page.on("request", _on_req)
    page.on("requestfinished", _on_done)
    page.on("requestfailed", _on_done)

    await page.goto(html_uri, wait_until="load")
    await page.wait_for_function("() => window.__tripsAnim && window.__tripsAnim.ready", timeout=60000)
    await page.evaluate("() => window.__tripsAnim.pause()")
    await page.wait_for_function("() => window.__tripsAnim.headReady", timeout=head_ready_timeout_ms)
    await page.add_script_tag(content=_CAM_HELPER)
    # The "Save as Image" widget (a camera icon, top-right) has no `id` in the rendered
    # DOM -- #SaveImageWidget never matched. Hide it by its actual deck.gl widget class.
    # The playback bar from draw_animated_map is for interactive viewing only.
    await page.add_style_tag(
        content=".deck-widget-save-image, #timeline-controls, #camera-panel { display: none !important; }"
    )
    return page, pending


_TILES_LOADED_JS = """() => {
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


async def _render_frames(page, pending, frames, clip, quality, capture, frame_dir, ext, progress):
    """Render an iterable of (index, t, viewState) to numbered files."""
    shot_kwargs = {"clip": clip, "type": quality.capture_format}
    if quality.capture_format == "jpeg":
        shot_kwargs["quality"] = quality.jpeg_quality
    for idx, t, vs in frames:
        await page.evaluate(
            """([t, vs]) => {
                const d = window.deckInstance;
                if (vs && d) d.setProps({ viewState: vs });
                window.__tripsAnim.renderAt(t);
                try { d.redraw && d.redraw('export'); } catch (e) {}
            }""",
            [t, vs],
        )
        # let deck issue tile requests for the new viewport, then drain them
        await page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        await _await_ready(page, pending, settle_timeout_ms=capture.settle_timeout_ms, settle_ms=capture.settle_ms)
        shot_kwargs["path"] = os.path.join(frame_dir, f"f_{idx:06d}.{ext}")
        await page.screenshot(**shot_kwargs)
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


def _raw_keyframes(camera: CameraOptions) -> list[dict]:
    """Keyframes given inline or via file. Validated before the browser launches; an empty
    list is allowed and means 'auto-derive a path from the data'."""
    if isinstance(camera, SequenceCamera):
        for segment in camera.segments:  # fail before the browser launches
            _raw_keyframes(segment.camera)
        return []
    if not isinstance(camera, KeyframesCamera):
        return []
    raw = [k.model_dump() if isinstance(k, BaseModel) else dict(k) for k in (camera.keyframes or [])]
    if not raw and isinstance(camera.source, KeyframesFromFile) and camera.source.keyframes_file:
        raw = _load_keyframes_file(camera.source.keyframes_file)
    if len(raw) == 1:
        raise ValueError(
            "camera='keyframes' needs at least 2 keyframes, got 1. "
            "Add more keyframes, or leave the list empty to auto-derive a camera path from the data."
        )
    return raw


def _camera_opts(
    camera: CameraOptions, kf_raw: list[dict], base_view: dict, width: int, height: int, fps: int = 30
) -> dict:
    """Options for window.__cam.path: the camera's fields (the page fills its own defaults),
    resolved keyframes, and the video size the framing is fitted to."""
    fields = {name: getattr(camera, name) for name in type(camera).model_fields}  # model_dump drops exclude=True
    source = fields.pop("source", None)
    fields.pop("keyframes", None)
    segments = fields.pop("segments", None)
    preset = fields.pop("type_")
    if segments is not None:  # sequence: every segment's camera gets its own options
        fields["segments"] = [
            {
                "until": seg.until,
                "camera": _camera_opts(seg.camera, _raw_keyframes(seg.camera), base_view, width, height, fps),
            }
            for seg in segments
        ]
    return {
        **fields,
        "preset": preset,
        "keyframes": _resolve_keyframes(kf_raw, base_view) if kf_raw else None,
        # keyframes name their subject in `source`; chase/route carry it as their own field
        "subject": source.subject if isinstance(source, KeyframesFromSubject) else fields.get("subject"),
        "auto_keyframe_count": 12,
        "width": width,
        "height": height,
        "fps": fps,
    }


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
    camera: CameraOptions = StaticCamera(),
    camera_settings: str | None = None,
    fps: int = 30,
    duration: DurationConfig = DurationConfig(),
    resolution: ResolutionOptions = PresetResolution(),
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    quality: VideoQuality = VideoQuality(),
    capture: CaptureOptions = CaptureOptions(),
    verbose: bool = True,
) -> str:
    """Async core of render_animation. ``await`` this directly inside a notebook if you prefer."""
    if camera_settings:
        camera = parse_camera_settings(camera_settings)
    width, height = _resolve_resolution(resolution)
    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(html_path)
    if output_dir:
        out_path = os.path.join(output_dir, os.path.basename(str(out_path)))
    ext = "jpg" if quality.capture_format == "jpeg" else "png"
    kf_raw = _raw_keyframes(camera)
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
        )

    _ensure_playwright_browsers()
    with tempfile.TemporaryDirectory(prefix="anim_frames_") as frame_dir:
        async with async_playwright() as p:
            browser = await _launch_browser(p, capture.gl)

            # One page reads the timeline and base view and computes the whole camera path.
            page0, pending0 = await open_page()
            span = await page0.evaluate("() => window.__cam.span() || window.__tripsAnim.span")
            base_view = await page0.evaluate("() => window.__cam.initialView()") or {}
            natural = await page0.evaluate("() => window.__tripsAnim.durationSec || 0")
            seconds = float(natural) if duration.auto and natural > 0 else float(duration.seconds)
            n_frames = max(1, int(round(fps * seconds)))
            workers = _auto_workers(n_frames, capture.gl) if capture.workers == "auto" else capture.workers
            log(
                f"span={span} duration={seconds:.2f}s frames={n_frames} camera={camera.type_} "
                f"gl={capture.gl} workers={workers}{' (auto)' if capture.workers == 'auto' else ''}"
            )

            t_lo, t_hi = span * start_frac, span * end_frac
            times = [t_lo + (t_hi - t_lo) * (i / (n_frames - 1) if n_frames > 1 else 1.0) for i in range(n_frames)]
            opts = _camera_opts(camera, kf_raw, base_view, width, height, fps)
            views = await page0.evaluate("([times, opts]) => window.__cam.path(times, opts)", [times, opts])

            canvas = await page0.query_selector("#deck-container canvas")
            if canvas is None:
                raise RuntimeError("deck-container canvas not found")
            clip = await canvas.bounding_box()

            done = {"n": 0}

            def progress(_idx):
                done["n"] += 1
                if done["n"] % 100 == 0 or done["n"] == n_frames:
                    el = time.time() - t_wall
                    log(f"{done['n']}/{n_frames} frames ({el:.1f}s, {done['n'] / max(el, 1e-6):.1f} fps)")

            # Contiguous chunks per worker page -> good tile-cache locality.
            frames = list(zip(range(n_frames), times, views))
            per = -(-n_frames // workers)
            chunks = [frames[i : i + per] for i in range(0, n_frames, per)]
            pages = [(page0, pending0)] + [await open_page() for _ in chunks[1:]]
            await asyncio.gather(
                *(
                    _render_frames(page, pending, chunk, clip, quality, capture, frame_dir, ext, progress)
                    for (page, pending), chunk in zip(pages, chunks)
                )
            )
            await browser.close()

        _encode_mp4(frame_dir, ext, fps, quality, out_path)

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
            default=StaticCamera(),
            discriminator="type_",
            title="Camera",
            description="Camera for the clip. Every option frames whatever is animating (trips, points, "
            "polygons, ...) at the video's size, with the map's pitch and bearing: Static (whole scene), Follow the "
            "action, Orbit, Fit everything so far, Cinematic fly-through, Keyframes (waypoints, or follow one "
            "subject), Fly to & around, Chase a subject, Subject tour, Event spotlight, Pan across, Route "
            "flythrough, or a Sequence of these.",
        ),
    ] = StaticCamera(),
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
            description="Output video resolution: a preset (720p/1080p/4K) or 'custom' width/height.",
        ),
    ] = PresetResolution(),
    start_frac: Annotated[
        float, AdvancedField(default=0.0, ge=0, le=1, description="Timeline fraction to start at (0 = beginning).")
    ] = 0.0,
    end_frac: Annotated[
        float, AdvancedField(default=1.0, ge=0, le=1, description="Timeline fraction to stop at (1 = end).")
    ] = 1.0,
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
        quality=quality,
        capture=capture,
        verbose=verbose,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running (Jupyter/IPython): run the coroutine on a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()

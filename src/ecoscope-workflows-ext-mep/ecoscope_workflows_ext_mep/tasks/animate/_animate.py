import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal
import imageio_ffmpeg
from wt_registry import register
from ecoscope.platform.annotations import AdvancedField
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field
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


class StaticCamera(BaseModel):
    """Holds one view that fits all the animated data for the whole clip."""

    model_config = ConfigDict(json_schema_extra={"title": "Static"})
    type_: Annotated[Literal["static"], Field(default="static", title="Camera")] = "static"
    zoom_offset: ZoomOffset = 0.0


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
    fit_padding: Annotated[int, AdvancedField(default=80, ge=0, description="Padding in pixels around the action.")] = (
        80
    )


class OrbitCamera(BaseModel):
    """Circles the center of all the animated data."""

    model_config = ConfigDict(json_schema_extra={"title": "Orbit"})
    type_: Annotated[Literal["orbit"], Field(default="orbit", title="Camera")] = "orbit"
    zoom_offset: ZoomOffset = 0.0
    rounds: Annotated[float, AdvancedField(default=1.0, gt=0, description="Full turns over the clip.")] = 1.0


class FitCamera(BaseModel):
    """Zooms to keep everything shown so far in frame."""

    model_config = ConfigDict(json_schema_extra={"title": "Fit everything so far"})
    type_: Annotated[Literal["fit"], Field(default="fit", title="Camera")] = "fit"
    zoom_offset: ZoomOffset = 0.0
    fit_padding: Annotated[int, AdvancedField(default=80, ge=0, description="Padding in pixels around the data.")] = 80


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
    fit_padding: Annotated[int, AdvancedField(default=80, ge=0, description="Padding in pixels around the action.")] = (
        80
    )


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
            description="Tilt on arrival (0 = straight down). Leave blank to use the " "map's pitch.",
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


CameraOptions = (
    StaticCamera | FollowCamera | OrbitCamera | FitCamera | CinematicCamera | KeyframesCamera | FlyAroundCamera
)
CameraConfig = Annotated[CameraOptions, Field(discriminator="type_")]


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
          tracks.push({ d: d, C: C, T: T });
          for (var k = 0; k < C.length; k++) all.push(C[k]);
          end = Math.max(end, T[T.length - 1]);
        } else {
          var t = rowTime(d), P = [];
          flatten(d.geometry && d.geometry.coordinates, P);
          if (!P.length || !isFinite(t)) return;
          rows.push({ d: d, t: t, P: P, w: s.window });
          for (var j = 0; j < P.length; j++) all.push(P[j]);
          end = Math.max(end, t);
        }
      });
    });
    var b = new Box(); all.forEach(function (p) { b.add(p); });
    var sz = 0; all.forEach(function (p) { sz += p[2] || 0; });
    SCENE = { tracks: tracks, rows: rows, box: b, meanZ: all.length ? sz / all.length : 0, end: end };
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
  function posAt(tr, t) {      // interpolated [lon, lat] of a track at time t (clamped)
    var C = tr.C, T = tr.T, last = T.length - 1;
    if (t <= T[0]) return C[0];
    if (t >= T[last]) return C[last];
    var j = Math.max(0, lowerBound(T, t) - 1), f = (T[j + 1] > T[j]) ? (t - T[j]) / (T[j + 1] - T[j]) : 0;
    return [C[j][0] + (C[j + 1][0] - C[j][0]) * f, C[j][1] + (C[j + 1][1] - C[j][1]) * f];
  }

  // What's on screen around t: track vertices from [t - w, t] + current positions, and the
  // rows each time-window layer is showing (its own window, or w for cumulative layers).
  function recentBox(t, w) {
    var S = scene(), b = new Box();
    S.tracks.forEach(function (tr) {
      if (t < tr.T[0] || t - w > tr.T[tr.T.length - 1]) return;     // not started / long finished
      for (var k = lowerBound(tr.T, t - w); k < tr.T.length && tr.T[k] <= t; k++) b.add(tr.C[k]);
      b.add(posAt(tr, t));
    });
    S.rows.forEach(function (r) {
      var from = t - (r.w == null ? w : Math.max(r.w, w));
      if (r.t >= from && r.t <= t) r.P.forEach(function (p) { b.add(p); });
    });
    return b;
  }
  // Everything shown up to t (tracks' traversed paths, rows that have appeared).
  function seenBox(t) {
    var S = scene(), b = new Box();
    S.tracks.forEach(function (tr) { for (var k = 0; k < tr.T.length && tr.T[k] <= t; k++) b.add(tr.C[k]); });
    S.rows.forEach(function (r) { if (r.t <= t) r.P.forEach(function (p) { b.add(p); }); });
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
    if (!a || !b || (a.longitude === b.longitude && a.latitude === b.latitude)) return null;
    var lat = (a.latitude + b.latitude) * 0.5 * Math.PI / 180;
    return Math.atan2((b.longitude - a.longitude) * Math.cos(lat), b.latitude - a.latitude) * 180 / Math.PI;
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
    u = Math.max(0, Math.min(1, u));
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
  // Time-ordered (t, lon, lat) samples for one subject: a matching track, else matching
  // rows of point/polygon layers. No subject -> the longest-running track.
  function subjectSamples(subject) {
    var S = scene(), key = (subject == null || subject === '') ? null : String(subject);
    var tr = null;
    if (key == null) {
      S.tracks.forEach(function (x) {
        if (!tr || x.T[x.T.length - 1] - x.T[0] > tr.T[tr.T.length - 1] - tr.T[0]) tr = x;
      });
    } else {
      tr = S.tracks.find(function (x) { return values(x.d).indexOf(key) >= 0; }) || null;
    }
    if (tr) return tr.T.map(function (t, k) { return { t: t, p: tr.C[k] }; });
    var rows = key == null ? [] : S.rows.filter(function (r) { return values(r.d).indexOf(key) >= 0; });
    return rows.sort(function (a, b) { return a.t - b.t; }).map(function (r) {
      var b = new Box(); r.P.forEach(function (p) { b.add(p); }); var c = b.center();
      return { t: r.t, p: [c.lon, c.lat] };
    });
  }
  function autoKeyframes(o, base) {
    var smp = subjectSamples(o.subject);
    if (smp.length < 2) return [];
    var n = o.auto_keyframe_count || 12, END = span() || 1, b = new Box();
    smp.forEach(function (x) { b.add(x.p); });
    var zoom = (fitBox(b, o) || base).zoom, out = [];
    for (var i = 0; i < n; i++) {
      var tt = smp[0].t + (smp[smp.length - 1].t - smp[0].t) * (i / (n - 1));
      var k = Math.max(1, lowerBound(smp.map(function (x) { return x.t; }), tt));
      var A = smp[k - 1], B = smp[Math.min(k, smp.length - 1)];
      var f = (B.t > A.t) ? (tt - A.t) / (B.t - A.t) : 0;
      out.push({ t: Math.min(1, tt / END), lon: A.p[0] + (B.p[0] - A.p[0]) * f, lat: A.p[1] + (B.p[1] - A.p[1]) * f,
                 zoom: zoom, pitch: base.pitch, bearing: base.bearing });
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Fly to & around (Google Maps 3D flyCameraTo + flyCameraAround)
  // ---------------------------------------------------------------------------
  // Google's `range` (camera-to-center metres) -> deck zoom. deck's MapView keeps the
  // camera 1.5 viewport-heights from the target, and the world is 512 * 2^zoom px wide.
  function zoomForRange(rangeM, lat, height) {
    return Math.log2(1.5 * height * 40075016.686 * Math.cos(lat * Math.PI / 180) / (512 * rangeM));
  }
  function flyAroundSetup(o, base) {
    var S = scene(), c = S.box.empty ? { lon: base.longitude, lat: base.latitude } : S.box.center();
    var lon = (o.lon != null) ? o.lon : c.lon, lat = (o.lat != null) ? o.lat : c.lat;
    var zoom = (o.range_m != null) ? zoomForRange(o.range_m, lat, o.height) + (o.zoom_offset || 0) : base.zoom;
    var end = { longitude: lon, latitude: lat, zoom: zoom,
                pitch: (o.tilt != null) ? o.tilt : base.pitch,
                bearing: base.bearing + shortestAngle(base.bearing, (o.heading != null) ? o.heading : base.bearing) };
    var FTI = (window.deck || {}).FlyToInterpolator;
    return { start: base, end: end, alt: (o.altitude != null) ? o.altitude : S.meanZ,
             fly: FTI ? new FTI({ curve: 1.414 }) : null,
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
    var smooth = Math.max(0, Math.min(1, o.follow_smoothing == null ? 0.25 : o.follow_smoothing));
    // follow_smoothing is per 1/30 s; rescale so the camera keeps up equally at any fps.
    var k = smooth > 0 ? 1 - Math.pow(1 - smooth, 30 / (o.fps || 30)) : 1;

    var KF = null, FA = null;
    if (preset === 'keyframes') {
      KF = (o.keyframes && o.keyframes.length >= 2) ? o.keyframes : autoKeyframes(o, base);
      KF.forEach(function (q) { if (q.zoom == null) q.zoom = base.zoom; });
    }
    if (preset === 'fly_around') FA = flyAroundSetup(o, base);

    var cur = null, prevTarget = null, cb = bearing, out = [], n = times.length;
    function follow(t) {                                   // smoothed framing of recent action
      var f = fitBox(recentBox(t, win), o);
      if (!f) return cur || base;
      cur = cur ? { longitude: cur.longitude + (f.longitude - cur.longitude) * k,
                    latitude:  cur.latitude  + (f.latitude  - cur.latitude ) * k,
                    zoom:      cur.zoom      + (f.zoom      - cur.zoom     ) * k } : f;
      return cur;
    }
    function faceTravel(target) {                          // smoothed heading of the camera's motion
      var h = travelHeading(prevTarget, target);
      prevTarget = target;
      if (h != null) cb += shortestAngle(cb, h) * Math.max(k, 0.12);
      return cb;
    }

    for (var i = 0; i < n; i++) {
      var t = times[i], prog = n > 1 ? i / (n - 1) : 1, vs;
      if (preset === 'follow') {
        var f = follow(t);
        vs = { longitude: f.longitude, latitude: f.latitude, zoom: f.zoom, pitch: pitch,
               bearing: o.heading_lock ? faceTravel(f) : bearing };
      } else if (preset === 'orbit') {
        vs = Object.assign({}, base, { bearing: (bearing + 360 * (o.rounds == null ? 1 : o.rounds) * prog) % 360 });
      } else if (preset === 'fit') {
        var fb = fitBox(seenBox(t), o) || base;
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
        if (intro > 0 && prog < intro) {                   // fly in from the whole-scene view
          var s = prog / intro, e = s * s * (3 - 2 * s);
          vs = { longitude: base.longitude + (vs.longitude - base.longitude) * e,
                 latitude:  base.latitude  + (vs.latitude  - base.latitude ) * e,
                 zoom:      base.zoom      + (vs.zoom      - base.zoom     ) * e,
                 pitch: pitch, bearing: vs.bearing };
        }
      } else if (preset === 'fly_around') {
        vs = flyAroundView(FA, prog);
      } else {
        vs = Object.assign({}, base);                      // static / unknown
      }
      out.push(vs);
    }
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
    await page.add_style_tag(content=".deck-widget-save-image, #timeline-controls { display: none !important; }")
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
    preset = fields.pop("type_")
    return {
        **fields,
        "preset": preset,
        "keyframes": _resolve_keyframes(kf_raw, base_view) if kf_raw else None,
        "subject": source.subject if isinstance(source, KeyframesFromSubject) else None,
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
            "subject), or Fly to & around.",
        ),
    ] = StaticCamera(),
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

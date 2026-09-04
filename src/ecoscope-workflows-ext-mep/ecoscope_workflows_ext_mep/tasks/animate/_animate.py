from __future__ import annotations
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
from wt_task.skip import SKIP_SENTINEL, SkipSentinel
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
            default=75.0,
            description="Video duration in seconds.",
        ),
    ] = 75.0


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


ResolutionConfig = Annotated[
    PresetResolution | CustomResolution,
    Field(discriminator="preset"),
]


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
    fall back to the scene's initial view.
    """

    lon: Annotated[float, Field(description="Longitude of the camera look-at point.")]
    lat: Annotated[float, Field(description="Latitude of the camera look-at point.")]
    t: Annotated[
        float | None,
        Field(
            default=None,
            description="Position on the clip timeline (0 = start, 1 = end). "
            "Any monotonically increasing numbers work (they are normalized to 0–1); "
            "omit everywhere for even spacing.",
        ),
    ] = None
    zoom: Annotated[float | None, Field(default=None, description="Zoom level at this keyframe.")] = None
    pitch: Annotated[float | None, Field(default=None, description="Camera tilt in degrees at this keyframe.")] = None
    bearing: Annotated[
        float | None, Field(default=None, description="Camera heading in degrees at this keyframe.")
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
      that define them (held flat past the ends), defaulting to the scene's
      initial view when no keyframe defines the channel at all.
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
        "zoom": base_view.get("zoom", 8),
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
    zoom: float | None = None,
    pitch: float | None = None,
    bearing_from_travel: bool = False,
    smooth_window: int = 3,
    span: float | None = None,
    name_col: str = "name",
) -> list[CameraKeyframe]:
    """Derive camera keyframes from a trajectory GeoDataFrame.

    Expects the same shape the animated TripsLayer is built from: one row per
    subject with a ``timestamps`` sequence and a (2D or Z) LineString whose
    vertices align with those timestamps.

    - ``subject``: row to follow — a value in ``name_col`` (or ``groupby_col``),
      a positional index, ``"all"`` for the mean position of every subject, or
      None to pick the longest-running track.
    - Keyframe ``t`` is ``timestamp / span`` where ``span`` defaults to the
      largest final timestamp in the gdf — i.e. the same span the animation
      plays over — so the camera is where the subject is *at that moment*.
      (If you render with start_frac/end_frac trims, pass span accordingly.)
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
    global_span = float(span) if span else max(float(T[-1]) for T, _ in tracks if len(T))
    if global_span <= 0:
        raise ValueError("could not determine a positive time span from the gdf")

    if isinstance(subject, str) and subject != "all":
        cols = [c for c in (name_col, "groupby_col") if c in gdf.columns]
        mask = None
        for c in cols:
            m = gdf[c].astype(str) == subject
            if m.any():
                mask = m
                break
        if mask is None:
            raise ValueError(f"subject {subject!r} not found in columns {cols}")
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
        pts = np.stack(
            [np.convolve(padded[:, 0], kernel, "valid"), np.convolve(padded[:, 1], kernel, "valid")], axis=1
        )

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
            t=float(min(1.0, max(0.0, sample_times[i] / global_span))),
            zoom=zoom,
            pitch=pitch,
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
            description="Subject to follow: a value from the 'name' (or 'groupby_col') column, 'all' for the group "
            "mean position, or empty for the longest-running track.",
        ),
    ] = None,
    n_keyframes: Annotated[
        int, AdvancedField(default=12, ge=2, description="Number of camera keyframes to sample along the track.")
    ] = 12,
    zoom: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(default=None, description="Zoom applied to every keyframe. Empty → the scene's initial zoom."),
    ] = None,
    pitch: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(default=None, description="Pitch applied to every keyframe. Empty → the scene's initial pitch."),
    ] = None,
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
        zoom=zoom,
        pitch=pitch,
        bearing_from_travel=bearing_from_travel,
        smooth_window=smooth_window,
    )


class StaticCamera(BaseModel):
    """Camera holds the scene's initial view for the whole clip.

    No motion, no overrides -- whatever view the map/animation was authored
    with is what renders. Pick one of the other camera types for movement.
    """

    type_: Literal["static"] = "static"


class FollowCamera(BaseModel):
    """Camera tracks a subject (or the group) in a flat, top-down-ish view.

    Longitude/latitude follow the tracked subject(s) every frame; pitch stays
    level (0) unless overridden and bearing stays fixed. For a tilted, 3D
    chase-cam that can also rotate with the subject's heading, use
    Follow3DCamera instead.
    """

    type_: Literal["follow"] = "follow"
    subject_index: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            exclude=True,
            description="Index of the subject to follow. Ignored when subjects='all'. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0
    subjects: Annotated[
        Literal["single", "all"],
        Field(
            default="all",
            exclude=True,
            description="'single' follows subject_index; 'all' follows the group's center and — unless a zoom "
            "override is set — smoothly zooms so every subject stays in frame. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = "all"
    zoom: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Zoom override. None -> the scene's initial zoom. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    pitch: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera tilt override in degrees. None -> flat (0). "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    bearing: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera heading override in degrees. None -> the scene's initial bearing. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    follow_smoothing: Annotated[
        float,
        Field(
            default=0.25,
            ge=0,
            le=1,
            exclude=True,
            description="Interpolation factor for camera movement (0 = instant snap, 1 = no lag). "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0.25
    zoom_boost: Annotated[
        float,
        Field(
            default=0.0,
            exclude=True,
            description="Zoom levels added on top of the base zoom. Positive = closer in. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0.0
    fit_padding: Annotated[
        int,
        Field(
            default=80,
            ge=0,
            exclude=True,
            description="Pixel padding used when subjects='all' to fit every subject in frame. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 80


class Follow3DCamera(BaseModel):
    """Camera tracks a subject (or the group) in a tilted, 3D chase-cam view.

    Like FollowCamera, but pitched by default and able to rotate its heading
    to match the tracked subject's direction of travel (heading_lock).
    """

    type_: Literal["follow_3d"] = "follow_3d"
    subject_index: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            exclude=True,
            description="Index of the subject to follow. Ignored when subjects='all'. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0
    subjects: Annotated[
        Literal["single", "all"],
        Field(
            default="all",
            exclude=True,
            description="'single' follows subject_index; 'all' follows the group's center and — unless a zoom "
            "override is set — smoothly zooms so every subject stays in frame. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = "all"
    zoom: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Zoom override. None -> the scene's initial zoom. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    pitch: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera tilt override in degrees. None -> the scene's initial pitch. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    bearing: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera heading override in degrees. Ignored when heading_lock is on. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    follow_smoothing: Annotated[
        float,
        Field(
            default=0.25,
            ge=0,
            le=1,
            exclude=True,
            description="Interpolation factor for camera movement (0 = instant snap, 1 = no lag). "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0.25
    zoom_boost: Annotated[
        float,
        Field(
            default=0.0,
            exclude=True,
            description="Zoom levels added on top of the base zoom. Positive = closer in. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0.0
    fit_padding: Annotated[
        int,
        Field(
            default=80,
            ge=0,
            exclude=True,
            description="Pixel padding used when subjects='all' to fit every subject in frame. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 80
    heading_lock: Annotated[
        bool,
        Field(
            default=False,
            exclude=True,
            description="Rotate the camera to match the tracked subject's travel direction. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = False


class OrbitCamera(BaseModel):
    """Camera circles the scene's centroid at a constant tilt.

    Not tied to any subject -- it orbits the mean position of every visited
    point in the scene.
    """

    type_: Literal["orbit"] = "orbit"
    zoom: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Zoom override. None -> the scene's initial zoom. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    pitch: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera tilt override in degrees. None -> 45. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    bearing: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Starting heading in degrees. None -> the scene's initial bearing. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    orbits: Annotated[
        float,
        Field(
            default=1.0,
            exclude=True,
            description="Number of full rotations to complete over the clip. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 1.0


class FitCamera(BaseModel):
    """Camera zooms to keep every point visited so far in frame.

    Zoom is always computed from the visited bounds (there's no zoom
    override); pitch and bearing can still be fixed.
    """

    type_: Literal["fit"] = "fit"
    subject_index: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            exclude=True,
            description="Index of the subject whose visited points to fit. Ignored when subjects='all'. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0
    subjects: Annotated[
        Literal["single", "all"],
        Field(
            default="all",
            exclude=True,
            description="'single' fits subject_index's visited points only; 'all' fits every subject's. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = "all"
    fit_padding: Annotated[
        int,
        Field(
            default=80,
            ge=0,
            exclude=True,
            description="Pixel padding around the fitted bounds. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 80
    pitch: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera tilt override in degrees. None -> the scene's initial pitch. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    bearing: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera heading override in degrees. None -> the scene's initial bearing. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None


class CinematicCamera(BaseModel):
    """Camera does a smooth fly-through: leads the subject, banks its bearing,
    and optionally flies in from altitude at the start.

    Faithful to the Mapbox "cinematic route" technique, expressed in deck.gl's
    MapView.
    """

    type_: Literal["cinematic"] = "cinematic"
    subject_index: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            exclude=True,
            description="Index of the subject to follow. Ignored when subjects='all'. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0
    subjects: Annotated[
        Literal["single", "all"],
        Field(
            default="all",
            exclude=True,
            description="'single' follows subject_index; 'all' follows the group's center and — unless a zoom "
            "override is set — smoothly zooms so every subject stays in frame. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = "all"
    zoom: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Zoom override. None -> the scene's initial zoom. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    pitch: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera tilt override in degrees. None -> 60 (deck's MapView caps ~60). "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    bearing: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Starting heading in degrees. None -> the scene's initial bearing. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    follow_smoothing: Annotated[
        float,
        Field(
            default=0.25,
            ge=0,
            le=1,
            exclude=True,
            description="Interpolation factor for the look-at point catching up to the leading edge. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0.25
    zoom_boost: Annotated[
        float,
        Field(
            default=0.0,
            exclude=True,
            description="Zoom levels added on top of the base zoom. Positive = closer in. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0.0
    fit_padding: Annotated[
        int,
        Field(
            default=80,
            ge=0,
            exclude=True,
            description="Pixel padding used when subjects='all' to fit every subject in frame. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 80
    lead_frac: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            le=1,
            exclude=True,
            description="Fraction of the total time span to look ahead of the subject. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0.0
    bearing_mode: Annotated[
        Literal["rotate", "heading", "fixed"],
        Field(
            default="rotate",
            exclude=True,
            description="'rotate' sweeps the bearing at a constant rate; 'heading' chases the subject's travel "
            "direction; 'fixed' holds the starting bearing. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = "rotate"
    rotate_deg: Annotated[
        float,
        Field(
            default=45.0,
            exclude=True,
            description="Total bearing sweep in degrees over the clip when bearing_mode is 'rotate'. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 45.0
    intro_frac: Annotated[
        float,
        Field(
            default=0.12,
            ge=0,
            le=1,
            exclude=True,
            description="Fraction of the clip used for the fly-in intro. 0 = no intro. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0.12
    intro_zoom_out: Annotated[
        float,
        Field(
            default=2.5,
            exclude=True,
            description="Zoom levels to pull back during the intro before flying into the scene. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 2.5


class KeyframesFromFile(BaseModel):
    """Camera path loaded from an uploaded waypoint file."""

    type_: Literal["file"] = "file"
    keyframes_file: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            description="Path to an uploaded keyframe file: a .json list of {lon, lat, t?, zoom?, pitch?, bearing?} "
            "objects, a .geojson of Point features (extras read from properties), or a .csv/.tsv with lon/lat "
            "columns.",
        ),
    ] = None


class KeyframesFromSubject(BaseModel):
    """Camera path auto-derived from the animated data by following one subject (or the group)."""

    type_: Literal["subject"] = "subject"
    subject: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            description="Which subject to follow -- a value from the 'name' (or 'groupby_col') column, a "
            "positional index (as digits, e.g. '2'), 'all' for the group's mean position, or empty for the "
            "longest-running track.",
        ),
    ] = None


KeyframeSource = Annotated[KeyframesFromSubject | KeyframesFromFile, Field(discriminator="type_")]


class KeyframesCamera(BaseModel):
    """Camera flies through a set of waypoints while the data animates.

    Pick a `source`: upload a waypoint file (KeyframesFromFile) or auto-derive a
    path by following a subject (KeyframesFromSubject) -- see also the
    derive_camera_keyframes task, which does the same derivation as an explicit,
    inspectable/editable list of keyframes. Alternatively, supply your own
    ``keyframes`` (a list of CameraKeyframe) programmatically -- these can be
    placed anywhere and are not tied to any subject's track, and take priority
    over `source` when non-empty.
    """

    type_: Literal["keyframes"] = "keyframes"
    keyframes: Annotated[
        list[CameraKeyframe] | SkipJsonSchema[None],
        Field(
            default=None,
            exclude=True,
            description="Camera waypoints. The camera flies through them in order while the animation plays. "
            "Each keyframe needs lon/lat; t (0–1 clip position), zoom, pitch, and bearing are optional and "
            "interpolated when omitted. Leave empty to use `source` instead. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = None
    source: Annotated[
        KeyframeSource,
        AdvancedField(
            default=KeyframesFromSubject(),
            description="How to build the camera path when `keyframes` is empty: upload a file, or auto-derive "
            "one by following a subject. Ignored when `keyframes` is provided directly.",
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
    zoom: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=12,
            description="Zoom applied to every auto-derived keyframe. None -> the scene's initial zoom. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 12
    pitch: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=45,
            description="Pitch applied to every auto-derived keyframe. None -> the scene's initial pitch. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 45
    bearing: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=0,
            description="Bearing applied to every auto-derived keyframe. None -> the scene's initial bearing. "
            "Not user-configurable; set via a workflow's spec.yaml if a non-default value is needed.",
        ),
    ] = 0


CameraConfig = Annotated[
    StaticCamera | FollowCamera | Follow3DCamera | OrbitCamera | FitCamera | CinematicCamera | KeyframesCamera,
    Field(discriminator="type_"),
]


# --- JS injected into each page: reads the scene data and builds the camera ----
# Parses window.jsonInput (the pydeck spec, a real global). `path(times, opts)`
# returns the full per-frame viewState array in a single call.
_CAM_HELPER = r"""
window.__cam = (function () {
  function trips() {
    var layers = (window.jsonInput && window.jsonInput.layers) || [];
    for (var i = 0; i < layers.length; i++) {
      var L = layers[i];
      if (L && (L["@@type"] === "TripsLayer" || ("currentTime" in L))) return L;
    }
    return null;
  }
  function feats() { var L = trips(); return (L && L.data) || []; }

  function headingDeg(C, j) {
    var last = C.length - 1;
    var a = C[Math.min(j, last)], b = C[Math.min(j + 1, last)];
    if (!a || !b) return 0;
    var lat = (a[1] + b[1]) * 0.5 * Math.PI / 180;
    var dx = (b[0] - a[0]) * Math.cos(lat), dy = (b[1] - a[1]);
    if (dx === 0 && dy === 0) return 0;
    return Math.atan2(dx, dy) * 180 / Math.PI;            // 0 = north, + = east
  }

  function headAt(t, idx) {
    var F = feats(); if (!F.length) return null;
    var f = F[Math.max(0, Math.min(idx || 0, F.length - 1))];
    var C = f.geometry && f.geometry.coordinates, T = f.timestamps;
    if (!C || !T || !C.length) return null;
    var last = T.length - 1;
    if (t <= T[0])    return { lon: C[0][0], lat: C[0][1], heading: headingDeg(C, 0) };
    if (t >= T[last]) return { lon: C[last][0], lat: C[last][1], heading: headingDeg(C, last - 1) };
    var lo = 0, hi = last;
    while (lo < hi) { var m = (lo + hi) >> 1; if (T[m] < t) lo = m + 1; else hi = m; }
    var j = Math.max(0, lo - 1), t0 = T[j], t1 = T[j + 1];
    var fr = (t1 > t0) ? (t - t0) / (t1 - t0) : 0, a = C[j], b = C[j + 1];
    return { lon: a[0] + (b[0] - a[0]) * fr, lat: a[1] + (b[1] - a[1]) * fr, heading: headingDeg(C, j) };
  }

  function span() {
    var F = feats(), s = 0;
    for (var i = 0; i < F.length; i++) { var T = F[i].timestamps; if (T && T.length) s = Math.max(s, T[T.length - 1]); }
    return s;
  }
  function initialView() { return (window.jsonInput && window.jsonInput.initialViewState) || null; }
  function centroid() {
    var F = feats(), sx = 0, sy = 0, n = 0;
    for (var i = 0; i < F.length; i++) {
      var C = F[i].geometry && F[i].geometry.coordinates; if (!C) continue;
      for (var k = 0; k < C.length; k++) { sx += C[k][0]; sy += C[k][1]; n++; }
    }
    return n ? { lon: sx / n, lat: sy / n } : { lon: 0, lat: 0 };
  }
  function boundsUpTo(t, idx, all) {
    var F = feats(); if (!F.length) return null;
    var rows = all ? F : [F[Math.max(0, Math.min(idx || 0, F.length - 1))]];
    var w = 180, s = 90, e = -180, nn = -90, any = false;
    for (var r = 0; r < rows.length; r++) {
      var f = rows[r]; if (!f) continue;
      var C = f.geometry && f.geometry.coordinates, T = f.timestamps; if (!C || !T) continue;
      for (var k = 0; k < C.length; k++) {
        if (T[k] > t) break; any = true;
        w = Math.min(w, C[k][0]); e = Math.max(e, C[k][0]); s = Math.min(s, C[k][1]); nn = Math.max(nn, C[k][1]);
      }
    }
    if (!any) {
      var h = all ? groupAt(t) : headAt(t, idx); if (!h) return null;
      w = e = h.lon; s = nn = h.lat;
    }
    return [[w, s], [e, nn]];
  }
  // Group state at time t: mean position of every subject plus the bounds that
  // contain them all. Subjects whose tracks have ended hold their last fix.
  function groupAt(t) {
    var F = feats(); if (!F.length) return null;
    var sx = 0, sy = 0, m = 0, w = 180, s = 90, e = -180, nn = -90;
    for (var j = 0; j < F.length; j++) {
      var h = headAt(t, j); if (!h) continue;
      sx += h.lon; sy += h.lat; m++;
      w = Math.min(w, h.lon); e = Math.max(e, h.lon);
      s = Math.min(s, h.lat); nn = Math.max(nn, h.lat);
    }
    return m ? { lon: sx / m, lat: sy / m, heading: 0, bounds: [[w, s], [e, nn]] } : null;
  }
  function fit(bounds, width, height, padding) {
    var deck = window.deck || window.deckgl || {}, VP = deck.WebMercatorViewport;
    if (!VP) return null;
    try {
      var vp = new VP({ width: width, height: height });
      var f = vp.fitBounds(bounds, { padding: padding == null ? 60 : padding });
      return { longitude: f.longitude, latitude: f.latitude, zoom: f.zoom };
    } catch (e) { return null; }
  }

  function shortestAngle(a, b) { var d = ((b - a + 180) % 360) - 180; return d <= -180 ? d + 360 : d; }

  // Centripetal-flavoured Catmull-Rom on one scalar channel.
  function catmullRom(p0, p1, p2, p3, u) {
    return 0.5 * ((2 * p1) + (-p0 + p2) * u
           + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
           + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u);
  }

  // Resolve the `subject` option to a feature index: a number is a positional
  // index; a string matches the feature's `name` (falling back to
  // `groupby_col`); null/undefined picks the longest-running track (the
  // feature whose last timestamp is greatest) -- mirrors keyframes_from_gdf's
  // `subject` argument on the Python side.
  function resolveSubjectIndex(F, subject) {
    if (typeof subject === 'number') return Math.max(0, Math.min(subject, F.length - 1));
    if (typeof subject === 'string') {
      if (/^-?\d+$/.test(subject)) return Math.max(0, Math.min(parseInt(subject, 10), F.length - 1));
      for (var i = 0; i < F.length; i++) {
        if (String(F[i].name) === subject || String(F[i].groupby_col) === subject) return i;
      }
      return 0; // no match -> fall back to the first feature
    }
    var best = 0, bestLast = -Infinity;
    for (var j = 0; j < F.length; j++) {
      var T = F[j].timestamps, last = (T && T.length) ? T[T.length - 1] : -Infinity;
      if (last > bestLast) { bestLast = last; best = j; }
    }
    return best;
  }

  // Fallback when the user picked the keyframes preset but supplied none:
  // sample a camera path from the trips data itself (t synced to the span).
  // mode 'single' follows one subject's track (`o.subject`: name, index, or
  // null for the longest-running track); mode 'all' follows the group center
  // and, unless a zoom override is given, zooms to keep EVERY subject in
  // frame (finished subjects hold their last position, so they stay shown).
  // Returns [] when the scene has no usable trips data.
  function autoKeyframes(o, n, zoom, pitch, bearing) {
    var F = feats(); if (!F.length) return [];
    var wantAll = (o.subject === 'all');
    var mode = (wantAll && F.length > 1) ? 'all' : 'single';
    var idx = mode === 'single' ? resolveSubjectIndex(F, o.subject) : 0;
    var SPAN = span() || 1;
    var out = [];
    if (mode === 'single') {
      var f = F[Math.max(0, Math.min(idx, F.length - 1))];
      var T = f && f.timestamps;
      if (!T || T.length < 2) return [];
      for (var i = 0; i < n; i++) {
        var tt = T[0] + (T[T.length - 1] - T[0]) * (i / (n - 1));
        var h = headAt(tt, idx); if (!h) continue;
        out.push({ t: Math.min(1, tt / SPAN), lon: h.lon, lat: h.lat,
                   zoom: zoom, pitch: pitch, bearing: bearing });
      }
      return out.length >= 2 ? out : [];
    }
    // mode === 'all'
    var t0 = Infinity;
    for (var j = 0; j < F.length; j++) {
      var Tj = F[j].timestamps;
      if (Tj && Tj.length) t0 = Math.min(t0, Tj[0]);
    }
    if (!isFinite(t0)) return [];
    for (var i = 0; i < n; i++) {
      var tt = t0 + (SPAN - t0) * (i / (n - 1));
      var g = groupAt(tt); if (!g) continue;
      var zk = zoom;                               // user override / base zoom
      if (o.zoom == null) {                        // no override -> fit the group
        var fz = fit(g.bounds, o.width, o.height, o.fit_padding);
        if (fz) zk = Math.min(fz.zoom, 14) + (o.zoom_boost || 0);  // cap: converged
      }                                            // subjects would over-zoom
      out.push({ t: Math.min(1, tt / SPAN), lon: g.lon, lat: g.lat,
                 zoom: zk, pitch: pitch, bearing: bearing });
    }
    return out.length >= 2 ? out : [];
  }

  // Interpolate the user-supplied keyframe path at clip-progress `prog` (0..1).
  // Keyframes arrive fully resolved from Python: sorted, t in [0,1], every
  // channel filled, bearings pre-unwrapped (so plain lerp rotates correctly).
  function keyframeView(K, prog, easing) {
    var hi = K.length - 1, s = 0;
    while (s + 1 < hi && K[s + 1].t <= prog) s++;
    var A = K[s], B = K[Math.min(s + 1, hi)];
    var u = (B.t > A.t) ? (prog - A.t) / (B.t - A.t) : 1;
    u = Math.max(0, Math.min(1, u));
    var ue = (easing === 'linear' || easing === 'spline') ? u : u * u * (3 - 2 * u); // smoothstep
    var lon, lat;
    if (easing === 'spline') {                       // Catmull-Rom through lon/lat
      var P0 = K[Math.max(0, s - 1)], P3 = K[Math.min(hi, s + 2)];
      lon = catmullRom(P0.lon, A.lon, B.lon, P3.lon, u);
      lat = catmullRom(P0.lat, A.lat, B.lat, P3.lat, u);
    } else {
      lon = A.lon + (B.lon - A.lon) * ue;
      lat = A.lat + (B.lat - A.lat) * ue;
    }
    return { longitude: lon, latitude: lat,
             zoom:    A.zoom    + (B.zoom    - A.zoom   ) * ue,
             pitch:   A.pitch   + (B.pitch   - A.pitch  ) * ue,
             bearing: A.bearing + (B.bearing - A.bearing) * ue };
  }

  // Build the entire per-frame viewState array in one pass.
  function path(times, o) {
    o = o || {};
    var preset = o.preset, idx = o.subject_index || 0;
    var base = initialView() || {};
    var zoom = ((o.zoom != null) ? o.zoom : (base.zoom != null ? base.zoom : 8))
               + (o.zoom_boost || 0);    // +1 ~ twice as close
    var pitch = (o.pitch != null) ? o.pitch : (base.pitch || 0);
    var bearing = (o.bearing != null) ? o.bearing : (base.bearing || 0);
    var smooth = Math.max(0, Math.min(1, o.follow_smoothing == null ? 0.25 : o.follow_smoothing));
    var orbits = (o.orbits == null) ? 1 : o.orbits;
    var cx = base.longitude, cy = base.latitude, cb = base.bearing || 0;
    // --- multi-subject support (subjects: 'all') -------------------------
    // follow/follow_3d/cinematic track the group's mean position instead of
    // one subject, and (when no zoom override is set) ease the zoom toward
    // whatever keeps EVERY subject in frame. fit aggregates all tracks.
    var multi = (o.subjects === 'all') && feats().length > 1;
    var cz = null, gpx = null, gpy = null;
    function lookAt(t) { return multi ? groupAt(t) : headAt(t, idx); }
    function groupZoom(g, fallback) {          // smoothed fit-zoom for the group
      if (!multi || o.zoom != null || !g || !g.bounds) return fallback;
      var f = fit(g.bounds, o.width, o.height, o.fit_padding);
      if (!f) return fallback;
      var tz = Math.min(f.zoom, 14) + (o.zoom_boost || 0);
      var k = Math.max(smooth, 0.1);
      cz = (cz == null) ? tz : cz + (tz - cz) * k;
      return cz;
    }
    function headingOf(h) {                    // travel direction; for the group,
      if (!multi) return h.heading;            // derived from the center's motion
      var hd = cb;
      if (gpx != null && (h.lon !== gpx || h.lat !== gpy)) {
        var latm = (gpy + h.lat) * 0.5 * Math.PI / 180;
        hd = Math.atan2((h.lon - gpx) * Math.cos(latm), h.lat - gpy) * 180 / Math.PI;
      }
      gpx = h.lon; gpy = h.lat;
      return hd;
    }
    // ---------------------------------------------------------------------
    var KF = o.keyframes || [];
    if (preset === 'keyframes' && KF.length < 2 && o.auto_keyframes) {
      KF = autoKeyframes(o, o.auto_keyframe_count || 12, zoom, pitch, bearing);
    }
    var out = [], n = times.length;
    for (var i = 0; i < n; i++) {
      var t = times[i], prog = n > 1 ? i / (n - 1) : 1, vs;
      if (preset === 'follow' || preset === 'follow_3d') {
        var h = lookAt(t);
        if (h) { var k = smooth > 0 ? smooth : 1;
                 cx = (cx == null) ? h.lon : cx + (h.lon - cx) * k;
                 cy = (cy == null) ? h.lat : cy + (h.lat - cy) * k; }
        vs = { longitude: cx, latitude: cy, zoom: groupZoom(h, zoom),
               pitch: preset === 'follow_3d' ? pitch : (o.pitch != null ? o.pitch : 0),
               bearing: bearing };
        if (preset === 'follow_3d' && o.heading_lock && h) {
          cb += shortestAngle(cb, headingOf(h)) * Math.max(smooth, 0.15); vs.bearing = cb;
        }
      } else if (preset === 'orbit') {
        var c = centroid();
        vs = { longitude: c.lon, latitude: c.lat, zoom: zoom,
               pitch: (o.pitch != null ? o.pitch : 45),
               bearing: (bearing + 360 * orbits * prog) % 360 };
      } else if (preset === 'fit') {
        var b = boundsUpTo(t, idx, multi), f = b ? fit(b, o.width, o.height, o.fit_padding) : null;
        vs = f ? { longitude: f.longitude, latitude: f.latitude, zoom: f.zoom, pitch: pitch, bearing: bearing }
               : Object.assign({}, base);
      } else if (preset === 'keyframes') {
        // User-authored camera path: fly through uploaded waypoints while the
        // data animation plays underneath, synced by clip progress. KF may be
        // auto-derived above; with <2 usable keyframes we hold the base view.
        vs = (KF.length >= 2) ? keyframeView(KF, prog, o.keyframe_easing || 'smooth')
                              : Object.assign({}, base);
      } else if (preset === 'cinematic') {
        // Faithful to the Mapbox "cinematic route" post, expressed in deck's
        // MapView: LERP the look-at toward the leading edge; pitch + zoom are
        // constant; bearing rotates at a CONSTANT rate, decoupled from the route
        // (their deliberate choice -- heading-locking got shaky on sharp turns).
        // deck's MapView places the camera behind/above the target from
        // (center, pitch, bearing, zoom), so their computeCameraPosition() trig
        // is handled for us.
        var SPAN = span();
        var lead = (o.lead_frac == null ? 0 : o.lead_frac) * SPAN;
        var here = lookAt(t);
        var target = (lead > 0) ? (lookAt(Math.min(SPAN, t + lead)) || here) : here;
        if (target) {                                    // lerp(prev, leadingEdge)
          var k = smooth > 0 ? smooth : 1;
          cx = (cx == null) ? target.lon : cx + (target.lon - cx) * k;
          cy = (cy == null) ? target.lat : cy + (target.lat - cy) * k;
        }
        var startB = (o.bearing != null) ? o.bearing : (base.bearing || 0);
        var mode = o.bearing_mode || 'rotate', brg;
        if (mode === 'heading' && here) {                // chase cam (their old way)
          cb += shortestAngle(cb, headingOf(here)) * Math.max(smooth, 0.12); brg = cb;
        } else if (mode === 'fixed') {
          brg = startB;
        } else {                                         // 'rotate' -- constant rate
          brg = startB + (o.rotate_deg == null ? 45 : o.rotate_deg) * prog;
        }
        var cpitch = (o.pitch != null ? o.pitch : 60);   // deck MapView caps ~60
        vs = { longitude: cx, latitude: cy, zoom: groupZoom(here, zoom), pitch: cpitch, bearing: brg };
        var introFrac = (o.intro_frac == null ? 0.12 : o.intro_frac);
        if (introFrac > 0 && prog < introFrac) {         // fly-in from altitude
          var s = prog / introFrac, e = s * s * (3 - 2 * s);   // smoothstep
          var c = centroid(), zo = (o.intro_zoom_out == null ? 2.5 : o.intro_zoom_out);
          var ov = { longitude: c.lon, latitude: c.lat, zoom: (zoom || 8) - zo, pitch: 35, bearing: vs.bearing };
          vs = { longitude: ov.longitude + (vs.longitude - ov.longitude) * e,
                 latitude:  ov.latitude  + (vs.latitude  - ov.latitude ) * e,
                 zoom:      ov.zoom       + (vs.zoom      - ov.zoom      ) * e,
                 pitch:     ov.pitch      + (vs.pitch     - ov.pitch     ) * e,
                 bearing:   vs.bearing };
        }
      } else { vs = Object.assign({}, base); }   // static / unknown
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
    await page.add_style_tag(content=".deck-widget-save-image { display: none !important; }")
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


async def _render_frames(
    page,
    pending,
    frames,
    clip,
    *,
    capture_format,
    jpeg_quality,
    settle_timeout_ms,
    settle_ms,
    frame_dir,
    ext,
    log,
    progress,
):
    """Render an iterable of (index, t, viewState) to numbered files."""
    shot_kwargs = {"clip": clip, "type": capture_format}
    if capture_format == "jpeg":
        shot_kwargs["quality"] = jpeg_quality
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
        await _await_ready(page, pending, settle_timeout_ms=settle_timeout_ms, settle_ms=settle_ms)
        shot_kwargs["path"] = os.path.join(frame_dir, f"f_{idx:06d}.{ext}")
        await page.screenshot(**shot_kwargs)
        progress(idx)


@register()
def configure_video_export(
    filename: Annotated[
        str,
        Field(description="Animated html file."),
    ] = "animated.html",
    enabled: Annotated[
        bool,
        Field(
            default=False,
            description="Render the animated map as a video file. "
            "When off, the video creation step is skipped entirely.",
        ),
    ] = False,
) -> str | SkipSentinel:
    """Return the output filename, or a skip sentinel when video export is disabled."""
    if not enabled:
        return SKIP_SENTINEL
    return filename


async def render_animation_async(
    html_path: str,
    output_dir: str | None = None,
    out_path: str = "animation.mp4",
    camera: CameraConfig = StaticCamera(),
    fps: int = 30,
    duration: DurationConfig = DurationConfig(),
    resolution: ResolutionConfig = PresetResolution(),
    device_scale_factor: int = 1,
    gl: str = "auto",  # "auto"/"angle" = GPU; "software" only if no GPU
    workers: int = 1,  # parallel browser pages
    capture_format: str = "jpeg",  # "jpeg" (fast) or "png" (lossless)
    jpeg_quality: int = 92,
    settle_ms: int = 30,
    settle_timeout_ms: int = 8000,
    head_ready_timeout_ms: int = 30000,
    crf: int = 18,
    x264_preset: str = "veryfast",  # ultrafast..medium..veryslow
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    verbose: bool = True,
) -> str:
    """Async core. ``await`` this directly inside a notebook if you prefer."""
    width, height = _resolve_resolution(resolution)
    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(html_path)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if output_dir:
        out_path = os.path.join(output_dir, os.path.basename(str(out_path)))
    else:
        out_path = str(out_path)
    workers = max(1, int(workers))
    ext = "jpg" if capture_format == "jpeg" else "png"
    html_uri = html_path.as_uri()

    # Validate keyframe input up front so bad input fails in milliseconds, not
    # after a browser launch. An *empty* list is not an error: we fall back to
    # auto-deriving a path from the animation's own data (subject_index track).
    kf_raw: list[dict] = []
    kf_auto = False
    if isinstance(camera, KeyframesCamera):
        kf_raw = [k.model_dump() if isinstance(k, BaseModel) else dict(k) for k in (camera.keyframes or [])]
        if not kf_raw and isinstance(camera.source, KeyframesFromFile) and camera.source.keyframes_file:
            kf_raw = _load_keyframes_file(camera.source.keyframes_file)
        if len(kf_raw) == 1:
            raise ValueError(
                "camera='keyframes' needs at least 2 keyframes, got 1. "
                "Add more keyframes, or leave the list empty to auto-derive a camera path from the data."
            )
        kf_auto = len(kf_raw) == 0

    def log(*a):
        if verbose:
            print(*a, file=sys.stderr, flush=True)

    t_wall = time.time()
    done = {"n": 0}

    def progress(_idx):
        done["n"] += 1
        if verbose and (done["n"] % 100 == 0 or done["n"] == n_frames):
            el = time.time() - t_wall
            log(f"[render] {done['n']}/{n_frames} frames  " f"({el:.1f}s, {done['n']/max(el,1e-6):.1f} fps)")

    _ensure_playwright_browsers()
    with tempfile.TemporaryDirectory(prefix="anim_frames_") as frame_dir:
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True, args=_launch_args(gl))
            except Exception as e:
                if "Executable doesn't exist" in str(e):
                    _ensure_playwright_browsers(force=True)
                    browser = await p.chromium.launch(headless=True, args=_launch_args(gl))
                else:
                    raise

            # One page to read span/base view and compute the whole camera path.
            page0, pending0 = await _prepare_page(
                browser,
                html_uri,
                width=width,
                height=height,
                device_scale_factor=device_scale_factor,
                head_ready_timeout_ms=head_ready_timeout_ms,
            )
            span = await page0.evaluate("() => window.__cam.span()") or await page0.evaluate(
                "() => window.__tripsAnim.span"
            )
            base_view = await page0.evaluate("() => window.__cam.initialView()") or {}
            log(f"[render] span={span}  base_view={base_view}  gl={gl}  workers={workers}")

            if duration.auto:
                nat = await page0.evaluate("() => (window.__tripsAnim && window.__tripsAnim.durationSec) || 0")
                if nat and nat > 0:
                    resolved_duration = float(nat)
                    log(f"[render] duration=auto -> {resolved_duration:.2f}s (animation's own length)")
                else:
                    resolved_duration = float(duration.seconds)
                    log(
                        f"[render] duration=auto, but the scene exposes no durationSec; "
                        f"using fallback {resolved_duration:.1f}s"
                    )
            else:
                resolved_duration = float(duration.seconds)
                log(f"[render] duration=fixed -> {resolved_duration:.2f}s")
            n_frames = max(1, int(round(fps * resolved_duration)))

            t_lo, t_hi = span * start_frac, span * end_frac
            times = [t_lo + (t_hi - t_lo) * (i / (n_frames - 1) if n_frames > 1 else 1.0) for i in range(n_frames)]

            resolved_keyframes = None
            keyframe_subject = (
                camera.source.subject
                if isinstance(camera, KeyframesCamera) and isinstance(camera.source, KeyframesFromSubject)
                else None
            )
            if isinstance(camera, KeyframesCamera):
                if kf_auto:
                    log(
                        "[render] camera=keyframes but no keyframes were provided; auto-deriving a camera "
                        f"path from the animation data (subject={keyframe_subject!r}). "
                        "Set `keyframes` or `source` to control the path."
                    )
                else:
                    resolved_keyframes = _resolve_keyframes(kf_raw, base_view)
                    log(
                        f"[render] camera=keyframes: {len(resolved_keyframes)} keyframes, "
                        f"easing={camera.keyframe_easing}"
                    )

            opts = {
                "preset": camera.type_,
                "subject_index": getattr(camera, "subject_index", 0),
                "subjects": getattr(camera, "subjects", "single"),
                "zoom": getattr(camera, "zoom", None),
                "pitch": getattr(camera, "pitch", None),
                "bearing": getattr(camera, "bearing", None),
                "follow_smoothing": getattr(camera, "follow_smoothing", 0.25),
                "heading_lock": getattr(camera, "heading_lock", False),
                "orbits": getattr(camera, "orbits", 1.0),
                "fit_padding": getattr(camera, "fit_padding", 80),
                "zoom_boost": getattr(camera, "zoom_boost", 0.0),
                "lead_frac": getattr(camera, "lead_frac", 0.0),
                "intro_frac": getattr(camera, "intro_frac", 0.12),
                "bearing_mode": getattr(camera, "bearing_mode", "rotate"),
                "rotate_deg": getattr(camera, "rotate_deg", 45.0),
                "intro_zoom_out": getattr(camera, "intro_zoom_out", 2.5),
                "keyframes": resolved_keyframes,
                "keyframe_easing": getattr(camera, "keyframe_easing", "smooth"),
                "subject": keyframe_subject,
                "auto_keyframes": kf_auto,
                "auto_keyframe_count": 12,
                "width": width,
                "height": height,
            }
            views = await page0.evaluate("([times, opts]) => window.__cam.path(times, opts)", [times, opts])

            canvas = await page0.query_selector("#deck-container canvas")
            if canvas is None:
                raise RuntimeError("deck-container canvas not found")
            box = await canvas.bounding_box()
            clip = {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}

            all_frames = list(zip(range(n_frames), times, views))

            common = dict(
                capture_format=capture_format,
                jpeg_quality=jpeg_quality,
                settle_timeout_ms=settle_timeout_ms,
                settle_ms=settle_ms,
                frame_dir=frame_dir,
                ext=ext,
                log=log,
                progress=progress,
            )

            if workers == 1:
                await _render_frames(page0, pending0, all_frames, clip, **common)
            else:
                # Contiguous chunks -> good tile-cache locality within each worker.
                chunks, per = [], (n_frames + workers - 1) // workers
                for w in range(workers):
                    sl = all_frames[w * per : (w + 1) * per]
                    if sl:
                        chunks.append(sl)

                async def run_chunk(chunk, page, pending):
                    await _render_frames(page, pending, chunk, clip, **common)

                tasks = [run_chunk(chunks[0], page0, pending0)]
                extra_pages = []
                for chunk in chunks[1:]:
                    pg, pend = await _prepare_page(
                        browser,
                        html_uri,
                        width=width,
                        height=height,
                        device_scale_factor=device_scale_factor,
                        head_ready_timeout_ms=head_ready_timeout_ms,
                    )
                    extra_pages.append(pg)
                    tasks.append(run_chunk(chunk, pg, pend))
                await asyncio.gather(*tasks)
                for pg in extra_pages:
                    await pg.close()

            await browser.close()

        # Assemble the numbered frame sequence into H.264.
        cmd = [
            ffmpeg_exe,
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            os.path.join(frame_dir, f"f_%06d.{ext}"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            x264_preset,
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            out_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed:\n" + proc.stderr.decode("utf-8", "ignore")[-2000:])

    log(f"[render] wrote {out_path}  ({time.time()-t_wall:.1f}s total)")
    return out_path


@register()
def render_animation(
    html_path: str,
    output_dir: str | None = None,
    out_path: str = "animation.mp4",
    camera: Annotated[
        CameraConfig,
        AdvancedField(
            default=StaticCamera(),
            description="Camera behavior for the clip. Pick a type — StaticCamera (initial view), FollowCamera/"
            "Follow3DCamera (tracks a subject), OrbitCamera (circles the scene), FitCamera (zooms to show all "
            "visited points), CinematicCamera (smooth fly-through), or KeyframesCamera (flies through waypoints) "
            "— then configure that type's fields.",
        ),
    ] = StaticCamera(),
    fps: Annotated[int, AdvancedField(default=30, gt=0, description="Output video frame rate.")] = 30,
    duration: Annotated[
        DurationConfig,
        AdvancedField(
            default=DurationConfig(),
            description="Video duration. 'auto' derives length from the animation's own playback time.",
        ),
    ] = DurationConfig(),
    resolution: Annotated[
        ResolutionConfig,
        AdvancedField(
            default=PresetResolution(),
            description="Output video resolution. Pick a common preset (720p/1080p/4K), or 'custom' to set an "
            "exact width/height.",
        ),
    ] = PresetResolution(),
    device_scale_factor: Annotated[
        int, AdvancedField(default=1, gt=0, description="Browser device pixel ratio. 2 = HiDPI/Retina output.")
    ] = 1,
    gl: Annotated[
        str,
        AdvancedField(
            default="auto",
            description="WebGL backend: 'auto' uses the GPU via ANGLE;",
        ),
    ] = "auto",
    workers: Annotated[
        int,
        AdvancedField(
            default=1,
            gt=0,
            description="Number of parallel browser pages for frame capture.",
        ),
    ] = 1,
    capture_format: Annotated[
        str,
        AdvancedField(
            default="jpeg", description="Per-frame image format: 'jpeg' (fast, small) or 'png' (lossless, larger)."
        ),
    ] = "jpeg",
    jpeg_quality: Annotated[
        int,
        AdvancedField(
            default=92, ge=1, le=100, description="JPEG quality (1–100). Only used when capture_format is 'jpeg'."
        ),
    ] = 92,
    settle_ms: Annotated[
        int,
        AdvancedField(
            default=30,
            ge=0,
            description="Extra milliseconds to wait after tiles are loaded before capturing each frame.",
        ),
    ] = 30,
    settle_timeout_ms: Annotated[
        int,
        AdvancedField(
            default=8000,
            gt=0,
            description="Maximum milliseconds to wait for tiles to finish loading per frame before giving up.",
        ),
    ] = 8000,
    head_ready_timeout_ms: Annotated[
        int,
        AdvancedField(
            default=30000,
            gt=0,
            description="Maximum milliseconds to wait for the 3D head model to load before starting capture.",
        ),
    ] = 30000,
    crf: Annotated[
        int,
        AdvancedField(
            default=18,
            ge=0,
            le=51,
            description="H.264 constant rate factor (0 = lossless, 51 = worst). Lower = better quality, larger file.",
        ),
    ] = 18,
    x264_preset: Annotated[
        str,
        AdvancedField(
            default="veryfast",
            description="x264 encoding speed preset (ultrafast → veryslow).",
        ),
    ] = "veryfast",
    start_frac: Annotated[
        float,
        AdvancedField(
            default=0.0,
            ge=0,
            le=1,
            description="Fraction of the animation timeline at which to start capturing (0 = beginning).",
        ),
    ] = 0.0,
    end_frac: Annotated[
        float,
        AdvancedField(
            default=1.0,
            ge=0,
            le=1,
            description="Fraction of the animation timeline at which to stop capturing (1 = end).",
        ),
    ] = 1.0,
    verbose: Annotated[
        bool,
        AdvancedField(default=True, description="Print progress and timing information to stderr during rendering."),
    ] = True,
) -> str:
    """Render an animated map HTML to an MP4 video file.

    No running event loop -> runs async core directly.
    A running loop (Jupyter/IPython) -> dispatches to a worker thread.
    """
    kwargs = dict(
        html_path=html_path,
        output_dir=output_dir,
        out_path=out_path,
        camera=camera,
        fps=fps,
        duration=duration,
        resolution=resolution,
        device_scale_factor=device_scale_factor,
        gl=gl,
        workers=workers,
        capture_format=capture_format,
        jpeg_quality=jpeg_quality,
        settle_ms=settle_ms,
        settle_timeout_ms=settle_timeout_ms,
        head_ready_timeout_ms=head_ready_timeout_ms,
        crf=crf,
        x264_preset=x264_preset,
        start_frac=start_frac,
        end_frac=end_frac,
        verbose=verbose,
    )

    def coro_factory():
        return render_animation_async(**kwargs)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    import concurrent.futures

    def _worker():
        return asyncio.run(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_worker).result()
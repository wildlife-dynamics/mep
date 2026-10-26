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


# Cameras are described the Google Maps 3D way (camera-position and altitude-modes docs): a center
# the camera looks at, its range (distance in metres), tilt, heading and lens angle (fov), and an
# altitude mode for the center's height. The page converts that to deck's view state.
AltitudeMode = Literal["clamp_to_ground", "relative_to_ground", "absolute"]

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
CameraAltitudeMode = Annotated[
    AltitudeMode,
    AdvancedField(
        default="clamp_to_ground",
        description="Height of the point the camera looks at: on the ground (clamp_to_ground), `altitude` metres "
        "above the ground (relative_to_ground), or at `altitude` metres above sea level (absolute). The ground is "
        "the height of the draped tracks nearest the center.",
    ),
]
CameraAltitude = Annotated[
    float | SkipJsonSchema[None],
    AdvancedField(
        default=None,
        title="Altitude (m)",
        description="Metres above the ground (relative_to_ground) or above sea level (absolute). "
        "Ignored by clamp_to_ground.",
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
    altitude_mode: CameraAltitudeMode = "clamp_to_ground"
    altitude: CameraAltitude = None


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
    altitude_mode: CameraAltitudeMode = "clamp_to_ground"
    altitude: CameraAltitude = None
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
    altitude_mode: CameraAltitudeMode = "clamp_to_ground"
    altitude: CameraAltitude = None
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
    altitude_mode: CameraAltitudeMode = "clamp_to_ground"
    altitude: CameraAltitude = None
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


CameraOptions = FixedCamera | FlyAroundCamera | FollowSubjectCamera | FollowActionCamera
CameraConfig = Annotated[CameraOptions, Field(discriminator="type_")]
_CAMERA_ADAPTER: TypeAdapter = TypeAdapter(CameraConfig)

# Fields the animated map's Camera panel edits with its own controls rather than a form input.
_PANEL_CUSTOM_FIELDS = ("type_",)


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
  // Fit a box to the video frame; boxes under MIN_BOX_DEG are widened around their center.
  // With the widget's tilt-aware fit (my-deck-widgets >= 0.15) the box fills the frame as seen
  // at the camera's tilt, heading and lens (path() puts them in o); otherwise it is fitted as if
  // looking straight down.
  function fitBox(b, o) {
    if (!b || b.empty) return null;
    var A = window.__tripsAnim, init = initialView() || {};
    if (A && typeof A.fitBounds === 'function') {
      try {
        var v = A.fitBounds([b.w, b.s, b.e, b.n], {
          width: o.width, height: o.height, padding: o.fit_padding == null ? 80 : o.fit_padding,
          pitch: o._pitch != null ? o._pitch : (init.pitch || 0),
          bearing: o._bearing != null ? o._bearing : (init.bearing || 0),
          fovy: o.fov || undefined, minDeg: MIN_BOX_DEG, maxZoom: MAX_ZOOM,
        });
        return { longitude: v.longitude, latitude: v.latitude, zoom: v.zoom };
      } catch (e) {}
    }
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
    return { longitude: c.lon, latitude: c.lat, zoom: z };
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
  // Google's `range` (camera-to-center metres) -> deck zoom. deck's MapView keeps the camera
  // `altitude` viewport-heights from the target (1.5 at its default ~37° field of view; a wider fov
  // brings it closer), and the world is 512 * 2^zoom px wide.
  function zoomForRange(rangeM, lat, height, fov) {
    var altitude = fov ? 0.5 / Math.tan(fov * Math.PI / 360) : 1.5;
    return Math.log2(altitude * height * EARTH_M * Math.cos(lat * Math.PI / 180) / (512 * rangeM));
  }
  function copies(v, n) { var out = []; for (var i = 0; i < n; i++) out.push(Object.assign({}, v)); return out; }

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
  // Fly to & around (Google Maps 3D flyCameraTo + flyCameraAround); Fixed view is
  // its end point held for the whole clip.
  // ---------------------------------------------------------------------------
  function flyAroundSetup(o, base) {
    var lon = (o.lon != null) ? o.lon : base.longitude, lat = (o.lat != null) ? o.lat : base.latitude;
    var zoom = (o.range_m != null) ? zoomForRange(o.range_m, lat, o.height, o.fov) : base.zoom;
    var end = { longitude: lon, latitude: lat, zoom: zoom, pitch: base.pitch, bearing: base.bearing };
    return { start: base, end: end, fly: flyer(),
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
               pitch: A.start.pitch, bearing: A.start.bearing };
    }
    var u = (A.frac < 1) ? (prog - A.frac) / (1 - A.frac) : 1;   // flyCameraAround
    return { longitude: A.end.longitude, latitude: A.end.latitude, zoom: A.end.zoom, pitch: A.end.pitch,
             bearing: (A.end.bearing + 360 * A.rounds * u) % 360 };
  }

  // ---------------------------------------------------------------------------
  // Follow a subject: behind it, facing where it is heading (or a set heading)
  // ---------------------------------------------------------------------------
  function chasePath(times, o, base) {
    var tr = subjectTrack(o.subject);
    if (!tr) return copies(base, times.length);
    var dt = Math.max((span() || 1) * 0.005, 1e-6);
    var dist = o.range_m || 800, tilt = o.tilt == null ? 60 : o.tilt;
    var kPos = rate(o.follow_smoothing, 0.3, o.fps), kTurn = rate(o.turn_smoothing, 0.15, o.fps);
    var cur = null, brg = null;
    return times.map(function (t) {
      var p = posAt(tr, t);
      if (o.heading != null) brg = o.heading;
      else {
        var h = headingLL(posAt(tr, t - dt), posAt(tr, t + dt));
        if (brg == null) brg = (h == null) ? base.bearing : h;
        else if (h != null) brg += shortestAngle(brg, h) * kTurn;
      }
      cur = cur ? [cur[0] + (p[0] - cur[0]) * kPos, cur[1] + (p[1] - cur[1]) * kPos,
                   cur[2] + ((p[2] || 0) - cur[2]) * kPos]
                : [p[0], p[1], p[2] || 0];
      return { longitude: cur[0], latitude: cur[1], zoom: zoomForRange(dist, cur[1], o.height, o.fov),
               pitch: tilt, bearing: brg, position: [0, 0, cur[2]] };
    });
  }

  // ---------------------------------------------------------------------------
  // Lens and altitude mode, applied to every camera's views
  // ---------------------------------------------------------------------------
  // fov -> the view's fovy. altitude_mode -> the height of the point looked at: the ground under it
  // (the draped tracks nearest it), `altitude` m above that, or `altitude` m above sea level.
  function applyLens(out, o) {
    var S = scene(), mode = o.altitude_mode || 'clamp_to_ground', alt = o.altitude;
    return out.map(function (v) {
      v = Object.assign({}, v);
      if (o.fov) v.fovy = o.fov;
      var ground = S.hasZ ? groundZ(v.longitude, v.latitude) : 0;
      var z = mode === 'absolute' ? (alt != null ? alt : ground)
            : mode === 'relative_to_ground' ? ground + (alt || 0)
            : (v.position ? v.position[2] : ground);          // clamp: a followed subject's own height
      v.position = [0, 0, z];
      return v;
    });
  }

  // ---------------------------------------------------------------------------
  // path(times, opts) -> one viewState per frame
  // ---------------------------------------------------------------------------
  function path(times, o) {
    o = o || {};
    var preset = o.preset, init = initialView() || {}, S = scene();
    // Tilt and heading: the camera's own, else the map's. Framing is fitted at these.
    var pitch = (o.tilt != null) ? o.tilt : (init.pitch || 0);
    var bearing = (o.heading != null) ? o.heading : (init.bearing || 0);
    o = Object.assign({}, o, { _pitch: pitch, _bearing: bearing });
    var whole = fitBox(S.box, o) || { longitude: init.longitude, latitude: init.latitude, zoom: init.zoom || 8 };
    var base = {
      longitude: whole.longitude, latitude: whole.latitude, zoom: whole.zoom, pitch: pitch, bearing: bearing
    };
    var END = span() || 1, win = (o.follow_window == null ? 0.1 : o.follow_window) * END;
    // follow_smoothing is per 1/30 s; rescale so the camera keeps up equally at any fps.
    var k = rate(o.follow_smoothing, 0.25, o.fps), keys = subjectKeys(o);
    var out = [], n = times.length;

    if (preset === 'chase') out = chasePath(times, o, base);
    else if (preset === 'follow') {
      var cur = null, prevTarget = null, cb = bearing;
      times.forEach(function (t) {
        var f = fitBox(recentBox(t, win, keys), o);     // smoothed framing of the recent action
        if (f) {
          cur = cur ? { longitude: cur.longitude + (f.longitude - cur.longitude) * k,
                        latitude:  cur.latitude  + (f.latitude  - cur.latitude ) * k,
                        zoom:      cur.zoom      + (f.zoom      - cur.zoom     ) * k } : f;
        }
        var v = cur || base;
        if (o.face_travel) {                             // smoothed heading of the camera's motion
          var h = travelHeading(prevTarget, v);
          prevTarget = v;
          if (h != null) cb += shortestAngle(cb, h) * Math.max(k, 0.12);
        }
        out.push({ longitude: v.longitude, latitude: v.latitude, zoom: v.zoom, pitch: pitch, bearing: cb });
      });
    } else if (preset === 'fly_around' || preset === 'fixed') {
      var flyOpts = preset === 'fixed' ? Object.assign({}, o, { fly_frac: 0, rounds: 0 }) : o;
      var FA = flyAroundSetup(flyOpts, base);
      for (var i = 0; i < n; i++) out.push(flyAroundView(FA, n > 1 ? i / (n - 1) : 1));
    } else {
      out = copies(base, n);
    }
    return applyLens(out, o);
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
    await page.add_script_tag(content=_CAM_HELPER)
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
    """Options for window.__cam.path: the camera's fields (the page fills its own defaults) and
    the video size the framing is fitted to."""
    fields = camera.model_dump()
    preset = fields.pop("type_")
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


class VideoSettings(BaseModel):
    """Every option of a video in one JSON object (render_video_from_settings). Anything left out
    takes render_animation's default."""

    model_config = ConfigDict(extra="forbid")
    camera: CameraConfig = FixedCamera()
    fps: int = Field(default=30, gt=0)
    duration: DurationConfig = DurationConfig()
    resolution: ResolutionConfig = PresetResolution()
    start_frac: float = Field(default=0.0, ge=0, le=1)
    end_frac: float = Field(default=1.0, ge=0, le=1)
    hold_start_s: float = Field(default=1.0, ge=0, le=30)
    hold_end_s: float = Field(default=2.0, ge=0, le=30)
    overlays: VideoOverlays = VideoOverlays()
    quality: VideoQuality = VideoQuality()
    capture: CaptureOptions = CaptureOptions()


def parse_video_settings(settings: str | dict | None) -> VideoSettings:
    """Video settings from a dict, JSON text, or the path of a .json file. A camera on its own (what
    the map's Camera panel copies or downloads, e.g. {"type_": "fixed", ...}) is accepted too, with
    every other option at its default. None or blank means every default."""
    if settings is None or (isinstance(settings, str) and not settings.strip()):
        return VideoSettings()
    if isinstance(settings, str):
        text = settings.strip()
        if not text.startswith(("{", "[")):  # a file path
            text = Path(remove_file_scheme(text)).read_text(encoding="utf-8")
        try:
            settings = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Video settings are not valid JSON: {e}") from e
    if not isinstance(settings, dict):
        raise ValueError("Video settings must be a JSON object.")
    if "type_" in settings:  # a camera alone
        settings = {"camera": settings}
    return VideoSettings.model_validate(settings)


@register()
def render_video_from_settings(
    html_path: Annotated[str, Field(description="Animated map HTML from draw_animated_map.")],
    settings: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            title="Video settings",
            description="JSON with the video's options -- camera, fps, duration, resolution, start_frac, end_frac, "
            "hold_start_s, hold_end_s, overlays, quality, capture -- as text or the path of a .json file. A camera "
            "on its own (the map's Camera panel settings) also works; anything left out takes its default. "
            "Leave blank for every default (a fixed view of the whole scene).",
        ),
    ] = None,
    output_dir: Annotated[str | SkipJsonSchema[None], Field(description="Directory for the video.")] = None,
    out_path: Annotated[
        str, Field(description="Video file name (or path when output_dir is unset).")
    ] = "animation.mp4",
    verbose: Annotated[bool, AdvancedField(default=True, description="Print progress to stderr.")] = True,
) -> str:
    """Render an animated map HTML to an MP4 video, with every option from one JSON."""
    video = parse_video_settings(settings)
    return _run_sync(
        render_animation_async(
            html_path,
            output_dir=output_dir,
            out_path=out_path,
            verbose=verbose,
            **{name: getattr(video, name) for name in VideoSettings.model_fields},
        )
    )

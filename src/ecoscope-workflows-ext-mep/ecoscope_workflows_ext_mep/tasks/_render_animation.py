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
from ecoscope_workflows_core.decorators import task
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

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
            description="Video duration in seconds. Used when auto is off, or as a fallback when auto cannot determine the length.",
        ),
    ] = 75.0


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
  function boundsUpTo(t, idx) {
    var F = feats(); var f = F[Math.max(0, Math.min(idx || 0, F.length - 1))]; if (!f) return null;
    var C = f.geometry && f.geometry.coordinates, T = f.timestamps; if (!C || !T) return null;
    var w = 180, s = 90, e = -180, nn = -90, any = false;
    for (var k = 0; k < C.length; k++) {
      if (T[k] > t) break; any = true;
      w = Math.min(w, C[k][0]); e = Math.max(e, C[k][0]); s = Math.min(s, C[k][1]); nn = Math.max(nn, C[k][1]);
    }
    if (!any) { var h = headAt(t, idx); if (!h) return null; w = e = h.lon; s = nn = h.lat; }
    return [[w, s], [e, nn]];
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
    var out = [], n = times.length;
    for (var i = 0; i < n; i++) {
      var t = times[i], prog = n > 1 ? i / (n - 1) : 1, vs;
      if (preset === 'follow' || preset === 'follow_3d') {
        var h = headAt(t, idx);
        if (h) { var k = smooth > 0 ? smooth : 1;
                 cx = (cx == null) ? h.lon : cx + (h.lon - cx) * k;
                 cy = (cy == null) ? h.lat : cy + (h.lat - cy) * k; }
        vs = { longitude: cx, latitude: cy, zoom: zoom,
               pitch: preset === 'follow_3d' ? pitch : (o.pitch != null ? o.pitch : 0),
               bearing: bearing };
        if (preset === 'follow_3d' && o.heading_lock && h) {
          cb += shortestAngle(cb, h.heading) * Math.max(smooth, 0.15); vs.bearing = cb;
        }
      } else if (preset === 'orbit') {
        var c = centroid();
        vs = { longitude: c.lon, latitude: c.lat, zoom: zoom,
               pitch: (o.pitch != null ? o.pitch : 45),
               bearing: (bearing + 360 * orbits * prog) % 360 };
      } else if (preset === 'fit') {
        var b = boundsUpTo(t, idx), f = b ? fit(b, o.width, o.height, o.fit_padding) : null;
        vs = f ? { longitude: f.longitude, latitude: f.latitude, zoom: f.zoom, pitch: pitch, bearing: bearing }
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
        var here = headAt(t, idx);
        var target = (lead > 0) ? (headAt(Math.min(SPAN, t + lead), idx) || here) : here;
        if (target) {                                    // lerp(prev, leadingEdge)
          var k = smooth > 0 ? smooth : 1;
          cx = (cx == null) ? target.lon : cx + (target.lon - cx) * k;
          cy = (cy == null) ? target.lat : cy + (target.lat - cy) * k;
        }
        var startB = (o.bearing != null) ? o.bearing : (base.bearing || 0);
        var mode = o.bearing_mode || 'rotate', brg;
        if (mode === 'heading' && here) {                // chase cam (their old way)
          cb += shortestAngle(cb, here.heading) * Math.max(smooth, 0.12); brg = cb;
        } else if (mode === 'fixed') {
          brg = startB;
        } else {                                         // 'rotate' -- constant rate
          brg = startB + (o.rotate_deg == null ? 45 : o.rotate_deg) * prog;
        }
        var cpitch = (o.pitch != null ? o.pitch : 60);   // deck MapView caps ~60
        vs = { longitude: cx, latitude: cy, zoom: zoom, pitch: cpitch, bearing: brg };
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
    base = ["--headless=new", "--ignore-gpu-blocklist", "--enable-unsafe-swapchains",
            "--no-sandbox", "--hide-scrollbars"]
    if gl == "software":
        return base + ["--use-gl=angle", "--use-angle=swiftshader"]
    if gl in ("angle", "auto"):
        return base + ["--use-gl=angle"]   # ANGLE picks Metal/GL/Vulkan -> real GPU
    return base


async def _prepare_page(browser, html_uri, *, width, height, device_scale_factor,
                        head_ready_timeout_ms):
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
    await page.wait_for_function(
        "() => window.__tripsAnim && window.__tripsAnim.ready", timeout=60000)
    await page.evaluate("() => window.__tripsAnim.pause()")
    await page.wait_for_function(
        "() => window.__tripsAnim.headReady", timeout=head_ready_timeout_ms)
    await page.add_script_tag(content=_CAM_HELPER)
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
            await page.wait_for_timeout(stable_ms)        # confirm it stays settled
            if pending["n"] <= 0:
                break
        if time.time() >= deadline:
            break
        await page.wait_for_timeout(15)
    if settle_ms:
        await page.wait_for_timeout(settle_ms)            # compositor cushion


async def _render_frames(page, pending, frames, clip, *, capture_format, jpeg_quality,
                         settle_timeout_ms, settle_ms, frame_dir, ext, log, progress):
    """Render an iterable of (index, t, viewState) to numbered files."""
    shot_kwargs = {"clip": clip, "type": capture_format}
    if capture_format == "jpeg":
        shot_kwargs["quality"] = jpeg_quality
    for (idx, t, vs) in frames:
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
        await page.evaluate(
            "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        await _await_ready(page, pending, settle_timeout_ms=settle_timeout_ms, settle_ms=settle_ms)
        shot_kwargs["path"] = os.path.join(frame_dir, f"f_{idx:06d}.{ext}")
        await page.screenshot(**shot_kwargs)
        progress(idx)

async def render_animation_async(
    html_path: str,
    output_dir: str | None = None,
    out_path: str = "animation.mp4",
    camera: Literal["static", "follow", "follow_3d", "orbit", "fit", "cinematic"] = "static",
    fps: int = 30,
    duration: DurationConfig = DurationConfig(),
    width: int = 1280,
    height: int = 720,
    device_scale_factor: int = 1,
    gl: str = "auto",                 # "auto"/"angle" = GPU; "software" only if no GPU
    workers: int = 1,                 # parallel browser pages
    capture_format: str = "jpeg",     # "jpeg" (fast) or "png" (lossless)
    jpeg_quality: int = 92,
    settle_ms: int = 30,
    settle_timeout_ms: int = 8000,
    head_ready_timeout_ms: int = 30000,
    crf: int = 18,
    x264_preset: str = "veryfast",    # ultrafast..medium..veryslow
    subject_index: int = 0,
    zoom: float | None = None,
    pitch: float | None = None,
    bearing: float | None = None,
    follow_smoothing: float = 0.25,
    zoom_boost: float = 0.0,
    heading_lock: bool = False,
    orbits: float = 1.0,
    fit_padding: int = 80,
    lead_frac: float = 0.0,
    bearing_mode: Literal["rotate", "heading", "fixed"] = "rotate",   # cinematic bearing mode
    rotate_deg: float = 45.0,         # cinematic "rotate": total bearing sweep over the clip
    intro_frac: float = 0.12,
    intro_zoom_out: float = 2.5,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    verbose: bool = True,
)->str:
    """Async core. ``await`` this directly inside a notebook if you prefer."""
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

    def log(*a):
        if verbose:
            print(*a, file=sys.stderr, flush=True)

    t_wall = time.time()
    done = {"n": 0}

    def progress(_idx):
        done["n"] += 1
        if verbose and (done["n"] % 10 == 0 or done["n"] == n_frames):
            el = time.time() - t_wall
            log(f"[render] {done['n']}/{n_frames} frames  "
                f"({el:.1f}s, {done['n']/max(el,1e-6):.1f} fps)")

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
                browser, html_uri, width=width, height=height,
                device_scale_factor=device_scale_factor,
                head_ready_timeout_ms=head_ready_timeout_ms)
            span = await page0.evaluate("() => window.__cam.span()") \
                or await page0.evaluate("() => window.__tripsAnim.span")
            base_view = await page0.evaluate("() => window.__cam.initialView()") or {}
            log(f"[render] span={span}  base_view={base_view}  gl={gl}  workers={workers}")

            if duration.auto:
                nat = await page0.evaluate(
                    "() => (window.__tripsAnim && window.__tripsAnim.durationSec) || 0")
                if nat and nat > 0:
                    resolved_duration = float(nat)
                    log(f"[render] duration=auto -> {resolved_duration:.2f}s (animation's own length)")
                else:
                    resolved_duration = float(duration.seconds)
                    log(f"[render] duration=auto, but the scene exposes no durationSec; "
                        f"using fallback {resolved_duration:.1f}s")
            else:
                resolved_duration = float(duration.seconds)
                log(f"[render] duration=fixed -> {resolved_duration:.2f}s")
            n_frames = max(1, int(round(fps * resolved_duration)))

            t_lo, t_hi = span * start_frac, span * end_frac
            times = [t_lo + (t_hi - t_lo) * (i / (n_frames - 1) if n_frames > 1 else 1.0)
                     for i in range(n_frames)]
            opts = {"preset": camera, "subject_index": subject_index, "zoom": zoom,
                    "pitch": pitch, "bearing": bearing, "follow_smoothing": follow_smoothing,
                    "heading_lock": heading_lock, "orbits": orbits, "fit_padding": fit_padding, "zoom_boost": zoom_boost,
                    "lead_frac": lead_frac, "intro_frac": intro_frac,
                    "bearing_mode": bearing_mode, "rotate_deg": rotate_deg,
                    "intro_zoom_out": intro_zoom_out, "width": width, "height": height}
            views = await page0.evaluate(
                "([times, opts]) => window.__cam.path(times, opts)", [times, opts])

            canvas = await page0.query_selector("#deck-container canvas")
            if canvas is None:
                raise RuntimeError("deck-container canvas not found")
            box = await canvas.bounding_box()
            clip = {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}

            all_frames = list(zip(range(n_frames), times, views))

            common = dict(capture_format=capture_format, jpeg_quality=jpeg_quality,
                          settle_timeout_ms=settle_timeout_ms, settle_ms=settle_ms,
                          frame_dir=frame_dir, ext=ext, log=log, progress=progress)

            if workers == 1:
                await _render_frames(page0, pending0, all_frames, clip, **common)
            else:
                # Contiguous chunks -> good tile-cache locality within each worker.
                chunks, per = [], (n_frames + workers - 1) // workers
                for w in range(workers):
                    sl = all_frames[w * per:(w + 1) * per]
                    if sl:
                        chunks.append(sl)

                async def run_chunk(chunk, page, pending):
                    await _render_frames(page, pending, chunk, clip, **common)

                tasks = [run_chunk(chunks[0], page0, pending0)]
                extra_pages = []
                for chunk in chunks[1:]:
                    pg, pend = await _prepare_page(
                        browser, html_uri, width=width, height=height,
                        device_scale_factor=device_scale_factor,
                        head_ready_timeout_ms=head_ready_timeout_ms)
                    extra_pages.append(pg)
                    tasks.append(run_chunk(chunk, pg, pend))
                await asyncio.gather(*tasks)
                for pg in extra_pages:
                    await pg.close()

            await browser.close()

        # Assemble the numbered frame sequence into H.264.
        cmd = [ffmpeg_exe, "-y", "-framerate", str(fps),
               "-start_number", "0", "-i", os.path.join(frame_dir, f"f_%06d.{ext}"),
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-preset", x264_preset, "-pix_fmt", "yuv420p",
               "-crf", str(crf), "-movflags", "+faststart", out_path]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed:\n" + proc.stderr.decode("utf-8", "ignore")[-2000:])

    log(f"[render] wrote {out_path}  ({time.time()-t_wall:.1f}s total)")
    return out_path


@task
def render_animation(
    html_path: str,
    output_dir: str | None = None,
    out_path: str = "animation.mp4",
    camera: Literal["static", "follow", "follow_3d", "orbit", "fit", "cinematic"] = "static",
    fps: int = 30,
    duration: DurationConfig = DurationConfig(),
    width: int = 1280,
    height: int = 720,
    device_scale_factor: int = 1,
    gl: str = "auto",
    workers: int = 1,
    capture_format: str = "jpeg",
    jpeg_quality: int = 92,
    settle_ms: int = 30,
    settle_timeout_ms: int = 8000,
    head_ready_timeout_ms: int = 30000,
    crf: int = 18,
    x264_preset: str = "veryfast",
    subject_index: int = 0,
    zoom: float | None = None,
    pitch: float | None = None,
    bearing: float | None = None,
    follow_smoothing: float = 0.25,
    zoom_boost: float = 0.0,
    heading_lock: bool = False,
    orbits: float = 1.0,
    fit_padding: int = 80,
    lead_frac: float = 0.0,
    bearing_mode: Literal["rotate", "heading", "fixed"] = "rotate",
    rotate_deg: float = 45.0,
    intro_frac: float = 0.12,
    intro_zoom_out: float = 2.5,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    verbose: bool = True,
) -> str:
    """Render an animated map HTML to an MP4 video file.

    No running event loop -> runs async core directly.
    A running loop (Jupyter/IPython) -> dispatches to a worker thread.
    """
    kwargs = dict(
        html_path=html_path, output_dir=output_dir, out_path=out_path, camera=camera, fps=fps,
        duration=duration, width=width,
        height=height, device_scale_factor=device_scale_factor, gl=gl,
        workers=workers, capture_format=capture_format, jpeg_quality=jpeg_quality,
        settle_ms=settle_ms, settle_timeout_ms=settle_timeout_ms,
        head_ready_timeout_ms=head_ready_timeout_ms, crf=crf,
        x264_preset=x264_preset, subject_index=subject_index, zoom=zoom,
        pitch=pitch, bearing=bearing, follow_smoothing=follow_smoothing,
        zoom_boost=zoom_boost, heading_lock=heading_lock, orbits=orbits,
        fit_padding=fit_padding, lead_frac=lead_frac, bearing_mode=bearing_mode,
        rotate_deg=rotate_deg, intro_frac=intro_frac, intro_zoom_out=intro_zoom_out,
        start_frac=start_frac, end_frac=end_frac, verbose=verbose,
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


"""render_animation's video options: resolution presets, overlays, opening/closing holds, resumable
frames and the bundled Chromium. The browser capture itself is not run here."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

from ecoscope_workflows_ext_mep.tasks.animate import _animate as anim
from ecoscope_workflows_ext_mep.tasks.animate._animate import (
    PresetResolution,
    VideoOverlays,
    _assemble_sequence,
    _frame_path,
    _frames_dir,
    _overlay_css,
    _render_key,
    _resolve_resolution,
    _use_bundled_chromium,
)


@pytest.mark.parametrize(
    "preset,size",
    [
        ("720p", (1280, 720)),
        ("1080p-vertical", (1080, 1920)),
        ("720p-vertical", (720, 1280)),
        ("1080p-square", (1080, 1080)),
    ],
)
def test_resolution_presets(preset, size):
    assert _resolve_resolution(PresetResolution(preset=preset)) == size


class TestOverlays:
    def test_everything_on_hides_only_the_interactive_controls(self):
        css = _overlay_css(VideoOverlays(), 720)
        hidden = css.split("{")[0]
        assert "#timeline-controls" in hidden and ".deck-widget-save-image" in hidden
        assert "legend" not in hidden and "scale" not in hidden and "north-arrow" not in hidden

    def test_switched_off_widgets_are_hidden(self):
        hidden = _overlay_css(VideoOverlays(legend=False, north_arrow=False, title=False), 720).split("{")[0]
        assert ".ecoscope-legend-widget" in hidden and ".ecoscope-north-arrow-widget" in hidden
        assert ".ecoscope-title-widget" in hidden and ".ecoscope-scale-widget" not in hidden

    def test_clock_corner_and_size_follow_the_video(self):
        css = _overlay_css(VideoOverlays(clock_position="bottom-left"), 1920)
        clock = css[css.index("#video-clock") :]
        assert "bottom: 38px" in clock and "left: 38px" in clock
        assert "font: 600 50px" in clock  # 2.6% of the height


class TestHolds:
    def _frames(self, tmp_path, n):
        for i in range(n):
            Path(_frame_path(str(tmp_path), i, "jpg")).write_text(str(i))

    def test_first_and_last_frames_are_held(self, tmp_path):
        self._frames(tmp_path, 3)
        seq = _assemble_sequence(str(tmp_path), "jpg", 3, n_hold_start=2, n_hold_end=3)
        got = [Path(seq, f).read_text() for f in sorted(os.listdir(seq))]
        assert got == ["0", "0", "0", "1", "2", "2", "2", "2"]

    def test_no_holds(self, tmp_path):
        self._frames(tmp_path, 2)
        seq = _assemble_sequence(str(tmp_path), "jpg", 2, 0, 0)
        assert sorted(os.listdir(seq)) == ["f_000000.jpg", "f_000001.jpg"]


class TestResume:
    def test_key_changes_with_the_page_and_the_settings(self, tmp_path):
        page = tmp_path / "map.html"
        page.write_text("<html>a</html>")
        k1 = _render_key(page, {"fps": 30})
        assert _render_key(page, {"fps": 30}) == k1
        assert _render_key(page, {"fps": 25}) != k1
        page.write_text("<html>b</html>")
        assert _render_key(page, {"fps": 30}) != k1

    def test_frames_live_beside_the_video(self, tmp_path):
        d = _frames_dir(str(tmp_path / "clip.mp4"), "abc123")
        assert Path(d) == tmp_path / ".clip-frames-abc123" and Path(d).is_dir()


class TestBundledChromium:
    def test_points_playwright_at_the_env_browsers(self, tmp_path, monkeypatch):
        (tmp_path / "share" / "ms-playwright" / "chromium_headless_shell-1234").mkdir(parents=True)
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        _use_bundled_chromium()
        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(tmp_path / "share" / "ms-playwright")

    def test_keeps_an_explicit_browsers_path(self, tmp_path, monkeypatch):
        (tmp_path / "share" / "ms-playwright" / "x").mkdir(parents=True)
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "/somewhere/else")
        _use_bundled_chromium()
        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == "/somewhere/else"

    def test_no_bundle_leaves_playwright_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        _use_bundled_chromium()
        assert "PLAYWRIGHT_BROWSERS_PATH" not in os.environ


def test_render_animation_exposes_the_new_options():
    import inspect

    params = inspect.signature(anim.render_animation).parameters
    assert {"hold_start_s", "hold_end_s", "overlays"} <= set(params)
    assert params["hold_start_s"].default == 1.0 and params["hold_end_s"].default == 2.0


class TestRenderVideoFromSettings:
    """The task's own options reach render_animation_async; the capture itself is stubbed out."""

    @pytest.fixture
    def rendered(self, monkeypatch):
        calls = []

        async def fake_render(html_path, **kwargs):
            calls.append(kwargs)
            return "animation.mp4"

        monkeypatch.setattr(anim, "render_animation_async", fake_render)
        return calls

    def test_defaults_are_a_fixed_view_of_the_whole_scene(self, rendered):
        anim.render_video_from_settings("map.html")
        (call,) = rendered
        assert call["camera"] == anim.FixedCamera() and call["fps"] == 30
        assert call["duration"] == anim.DurationConfig() and call["overlays"] == VideoOverlays()
        assert call["quality"] == anim.VideoQuality() and call["capture"] == anim.CaptureOptions()

    def test_video_options_are_passed_through(self, rendered):
        anim.render_video_from_settings(
            "map.html",
            camera_settings=anim.FollowSubjectCamera(subject="Big Guy", range_m=2500),
            fps=24,
            duration=anim.DurationConfig(auto=False, seconds=20),
            overlays=VideoOverlays(legend=False),
            quality=anim.VideoQuality(crf=20),
            capture=anim.CaptureOptions(resume=False),
        )
        (call,) = rendered
        assert (call["camera"].type_, call["camera"].subject, call["camera"].range_m) == ("chase", "Big Guy", 2500)
        assert call["fps"] == 24 and call["duration"].seconds == 20 and not call["overlays"].legend
        assert call["quality"].crf == 20 and call["capture"].resume is False

    def test_camera_panel_file(self, rendered, tmp_path):
        f = tmp_path / "camera-settings.json"
        f.write_text('{"type_": "fly_around", "rounds": 2}')
        anim.render_video_from_settings("map.html", camera_settings=anim.CameraFromSettings(settings=str(f)))
        assert rendered[0]["camera"] == anim.FlyAroundCamera(rounds=2)

    def test_pasted_camera_panel_settings(self, rendered):
        settings = anim.CameraFromSettings(settings='{"type_": "chase", "subject": "Amani"}')
        anim.render_video_from_settings("map.html", camera_settings=settings)
        assert rendered[0]["camera"] == anim.FollowSubjectCamera(subject="Amani")

    @pytest.mark.parametrize("text", ['{"type_": "orbit"}', '{"type_": "fixed", "tilt": 85}', "{bad"])
    def test_invalid_camera_panel_settings_are_rejected(self, rendered, text):
        with pytest.raises(ValueError):
            anim.render_video_from_settings("map.html", camera_settings=anim.CameraFromSettings(settings=text))
        assert not rendered

    def test_camera_field_offers_the_cameras_and_panel_settings(self):
        from pydantic import TypeAdapter

        param = inspect.signature(anim.render_video_from_settings).parameters["camera_settings"]
        adapter = TypeAdapter(param.annotation)
        assert adapter.validate_python({"type_": "keyframes"}) == anim.KeyframesCamera()
        assert adapter.validate_python({"type_": "settings", "settings": "{}"}).settings == "{}"
        mapping = adapter.json_schema()["discriminator"]["mapping"]
        assert set(mapping) == {"fixed", "fly_around", "chase", "follow", "keyframes", "settings"}

    def test_video_options_are_advanced_fields(self):
        params = inspect.signature(anim.render_video_from_settings).parameters
        for name in ("camera_settings", "fps", "duration", "overlays", "quality", "capture"):
            assert params[name].annotation.__metadata__[0].json_schema_extra["ecoscope:advanced"] is True

"""render_animation's video options: resolution presets, overlays, opening/closing holds, resumable
frames and the bundled Chromium. The browser capture itself is not run here."""

from __future__ import annotations

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


class TestVideoSettings:
    def test_camera_panel_file_alone(self, tmp_path):
        from ecoscope_workflows_ext_mep.tasks.animate._animate import FixedCamera, parse_video_settings

        f = tmp_path / "camera-settings.json"
        f.write_text('{"type_": "fixed", "altitude_mode": "clamp_to_ground"}')
        video = parse_video_settings(str(f))
        assert video.camera == FixedCamera() and video.fps == 30 and video.hold_end_s == 2.0

    def test_full_settings(self):
        from ecoscope_workflows_ext_mep.tasks.animate._animate import parse_video_settings

        video = parse_video_settings(
            '{"camera": {"type_": "chase", "subject": "Big Guy", "range_m": 2500}, "fps": 24,'
            ' "resolution": {"preset": "1080p-vertical"}, "duration": {"auto": false, "seconds": 20},'
            ' "overlays": {"legend": false, "clock_position": "bottom-left"}, "hold_start_s": 0}'
        )
        assert (video.camera.type_, video.camera.subject, video.camera.range_m) == ("chase", "Big Guy", 2500)
        assert video.fps == 24 and video.resolution.preset == "1080p-vertical" and video.duration.seconds == 20
        assert not video.overlays.legend and video.hold_start_s == 0

    def test_dict_input(self):
        from ecoscope_workflows_ext_mep.tasks.animate._animate import parse_video_settings

        assert parse_video_settings({"fps": 60}).fps == 60

    @pytest.mark.parametrize(
        "text", ['{"fps": 0}', '{"frame_rate": 30}', '{"camera": {"type_": "orbit"}}', "[1, 2]", "{bad"]
    )
    def test_invalid_settings_are_rejected(self, text):
        from ecoscope_workflows_ext_mep.tasks.animate._animate import parse_video_settings

        with pytest.raises(ValueError):
            parse_video_settings(text)

    def test_every_render_option_is_a_setting(self):
        import inspect

        from ecoscope_workflows_ext_mep.tasks.animate._animate import VideoSettings, render_animation_async

        params = set(inspect.signature(render_animation_async).parameters)
        assert set(VideoSettings.model_fields) == params - {
            "html_path",
            "output_dir",
            "out_path",
            "camera_settings",
            "verbose",
        }


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_blank_settings_mean_every_default(blank):
    from ecoscope_workflows_ext_mep.tasks.animate._animate import VideoSettings, parse_video_settings

    assert parse_video_settings(blank) == VideoSettings()


def test_settings_is_an_advanced_field_with_a_blank_default():
    import inspect

    from ecoscope_workflows_ext_mep.tasks.animate._animate import render_video_from_settings

    param = inspect.signature(render_video_from_settings).parameters["settings"]
    assert param.default is None
    field = param.annotation.__metadata__[0]
    assert field.json_schema_extra["ecoscope:advanced"] is True

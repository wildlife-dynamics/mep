"""Tests for the camera settings round trip between draw_animated_map's Camera panel and
render_animation: the panel's form schema and parsing the settings it copies or downloads."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ecoscope_workflows_ext_mep.tasks.animate._animate import (
    ChaseCamera,
    CinematicCamera,
    FollowCamera,
    KeyframesCamera,
    SequenceCamera,
    StaticCamera,
    _camera_opts,
    _raw_keyframes,
    camera_panel_schema,
    parse_camera_settings,
)


def _camera(schema, type_):
    return next(c for c in schema if c["type"] == type_)


def _field(camera, name):
    return next(f for f in camera["fields"] if f["name"] == name)


class TestCameraPanelSchema:
    def test_lists_every_camera_with_its_title(self):
        schema = camera_panel_schema()
        assert [c["type"] for c in schema] == [
            "static",
            "follow",
            "orbit",
            "fit",
            "cinematic",
            "keyframes",
            "fly_around",
            "chase",
            "tour",
            "spotlight",
            "pan",
            "route",
            "sequence",
        ]
        assert _camera(schema, "follow")["title"] == "Follow the action"
        assert _camera(schema, "follow")["description"].startswith("Follows the action")

    def test_fields_carry_defaults_limits_and_types(self):
        follow = _camera(camera_panel_schema(), "follow")
        window = _field(follow, "follow_window")
        assert window["type"] == "number"
        assert window["default"] == FollowCamera().follow_window
        assert (window["min"], window["max"]) == (0, 1)
        assert _field(follow, "heading_lock")["type"] == "boolean"
        assert _field(follow, "fit_padding")["type"] == "integer"

    def test_literals_become_option_lists(self):
        cinematic = _camera(camera_panel_schema(), "cinematic")
        mode = _field(cinematic, "bearing_mode")
        assert mode["type"] == "enum"
        assert mode["options"] == ["rotate", "heading", "fixed"]
        assert mode["default"] == CinematicCamera().bearing_mode

    def test_optional_values_default_to_auto(self):
        fly = _camera(camera_panel_schema(), "fly_around")
        assert _field(fly, "lon")["default"] is None
        assert _field(fly, "tilt")["max"] == 80

    def test_keyframe_list_and_source_are_left_to_the_panel(self):
        names = {f["name"] for f in _camera(camera_panel_schema(), "keyframes")["fields"]}
        assert names == {"zoom_offset", "keyframe_easing", "min_height_m"}

    def test_sequence_segments_are_left_to_the_panel(self):
        names = {f["name"] for f in _camera(camera_panel_schema(), "sequence")["fields"]}
        assert names == {"blend_frac", "min_height_m"}

    def test_subject_lists_are_arrays(self):
        schema = camera_panel_schema()
        assert _field(_camera(schema, "tour"), "subjects")["type"] == "array"
        assert _field(_camera(schema, "follow"), "subjects")["type"] == "array"

    def test_every_camera_has_min_height(self):
        for camera in camera_panel_schema():
            assert _field(camera, "min_height_m")["title"] == "Min height (m)"

    def test_schema_is_json_serialisable(self):
        json.dumps(camera_panel_schema())


class TestParseCameraSettings:
    def test_pasted_text(self):
        camera = parse_camera_settings('{"type_": "follow", "follow_window": 0.2, "heading_lock": true}')
        assert isinstance(camera, FollowCamera)
        assert camera.follow_window == 0.2
        assert camera.heading_lock is True

    def test_downloaded_file(self, tmp_path):
        path = tmp_path / "camera-settings.json"
        path.write_text(
            json.dumps(
                {
                    "type_": "keyframes",
                    "zoom_offset": 0,
                    "keyframe_easing": "spline",
                    "keyframes": [
                        {"t": 0, "lon": 36.8, "lat": -1.3, "zoom": 12, "pitch": 30, "bearing": 10},
                        {"t": 0.8, "lon": 36.95, "lat": -1.15, "zoom": 11, "pitch": 50, "bearing": 350},
                    ],
                }
            )
        )
        camera = parse_camera_settings(f"  {path}\n")
        assert isinstance(camera, KeyframesCamera)
        assert camera.keyframe_easing == "spline"
        assert [k.bearing for k in camera.keyframes] == [10, 350]

    def test_file_uri(self, tmp_path):
        path = tmp_path / "camera-settings.json"
        path.write_text('{"type_": "static", "zoom_offset": 1.5}')
        camera = parse_camera_settings(path.as_uri())
        assert camera == StaticCamera(zoom_offset=1.5)

    def test_keyframes_without_list_follow_a_subject(self):
        camera = parse_camera_settings('{"type_": "keyframes", "source": {"type_": "subject", "subject": "Amani"}}')
        assert camera.keyframes is None
        assert camera.source.subject == "Amani"

    def test_invalid_json_is_reported(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_camera_settings('{"type_": "follow",')

    def test_out_of_range_values_are_rejected(self):
        with pytest.raises(ValidationError):
            parse_camera_settings('{"type_": "follow", "follow_window": 3}')

    def test_unknown_camera_is_rejected(self):
        with pytest.raises(ValidationError):
            parse_camera_settings('{"type_": "drone"}')


class TestSequenceCamera:
    SETTINGS = {
        "type_": "sequence",
        "blend_frac": 0.1,
        "segments": [
            {"until": 0.3, "camera": {"type_": "chase", "subject": "Amani", "tilt": 70}},
            {
                "until": 0.8,
                "camera": {
                    "type_": "keyframes",
                    "keyframes": [
                        {"t": 0, "lon": 36.8, "lat": -1.3, "bearing": 10},
                        {"t": 1, "lon": 36.9, "lat": -1.2, "bearing": 350},
                    ],
                },
            },
            {"until": 1, "camera": {"type_": "fit", "subjects": ["Amani", "Baraka"]}},
        ],
    }

    def test_parses_nested_cameras(self):
        camera = parse_camera_settings(json.dumps(self.SETTINGS))
        assert isinstance(camera, SequenceCamera)
        assert [s.camera.type_ for s in camera.segments] == ["chase", "keyframes", "fit"]
        assert isinstance(camera.segments[0].camera, ChaseCamera)
        assert camera.segments[2].camera.subjects == ["Amani", "Baraka"]

    def test_options_nest_each_segment_camera(self):
        camera = parse_camera_settings(json.dumps(self.SETTINGS))
        opts = _camera_opts(camera, _raw_keyframes(camera), {"pitch": 0, "bearing": 0}, 1280, 720, 30)
        assert opts["preset"] == "sequence"
        assert opts["blend_frac"] == 0.1
        chase, keyframes, fit = (s["camera"] for s in opts["segments"])
        assert [s["until"] for s in opts["segments"]] == [0.3, 0.8, 1]
        assert (chase["preset"], chase["subject"], chase["tilt"]) == ("chase", "Amani", 70)
        assert (chase["width"], chase["height"], chase["fps"]) == (1280, 720, 30)
        # keyframes are resolved per segment: bearings unwrapped for the short turn
        assert [k["bearing"] for k in keyframes["keyframes"]] == [10, -10]
        assert fit["subjects"] == ["Amani", "Baraka"]

    def test_bad_keyframes_inside_a_segment_fail_early(self):
        camera = parse_camera_settings(
            '{"type_": "sequence", "segments": '
            '[{"camera": {"type_": "keyframes", "keyframes": [{"lon": 1, "lat": 2}]}}]}'
        )
        with pytest.raises(ValueError, match="at least 2 keyframes"):
            _raw_keyframes(camera)

    def test_sequences_cannot_nest(self):
        with pytest.raises(ValidationError):
            parse_camera_settings(
                '{"type_": "sequence", "segments": [{"camera": {"type_": "sequence", "segments": []}}]}'
            )

    def test_needs_a_segment(self):
        with pytest.raises(ValidationError):
            parse_camera_settings('{"type_": "sequence", "segments": []}')


class TestNewCameras:
    @pytest.mark.parametrize(
        "settings",
        [
            {"type_": "chase", "subject": "Amani", "distance_m": 500, "speed_zoom": 1.5},
            {"type_": "tour", "subjects": ["Amani", "Baraka"], "transition_frac": 0.3},
            {"type_": "spotlight", "layer": "layer-1", "between": "follow", "event_zoom": 14},
            {"type_": "pan", "direction": "diagonal", "easing": "linear"},
            {"type_": "route", "subject": "Baraka", "look_ahead_m": 0},
            {"type_": "follow", "subjects": ["Amani"], "speed_zoom": 2, "min_height_m": 400},
        ],
    )
    def test_round_trip(self, settings):
        camera = parse_camera_settings(json.dumps(settings))
        assert camera.model_dump(include=set(settings)) == settings

    def test_chase_subject_reaches_the_page(self):
        camera = parse_camera_settings('{"type_": "chase", "subject": "Amani"}')
        assert _camera_opts(camera, [], {}, 1280, 720)["subject"] == "Amani"

    def test_tilt_is_limited(self):
        with pytest.raises(ValidationError):
            parse_camera_settings('{"type_": "chase", "tilt": 85}')

    def test_unknown_pan_direction_is_rejected(self):
        with pytest.raises(ValidationError):
            parse_camera_settings('{"type_": "pan", "direction": "sideways"}')

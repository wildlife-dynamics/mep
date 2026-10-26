"""Tests for the camera settings round trip between draw_animated_map's Camera panel and
render_animation: the panel's form schema and parsing the settings it copies or downloads."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ecoscope_workflows_ext_mep.tasks.animate._animate import (
    FixedCamera,
    FlyAroundCamera,
    FollowActionCamera,
    FollowSubjectCamera,
    _camera_opts,
    camera_panel_schema,
    parse_camera_settings,
)


def _camera(schema, type_):
    return next(c for c in schema if c["type"] == type_)


def _field(camera, name):
    return next(f for f in camera["fields"] if f["name"] == name)


class TestCameraPanelSchema:
    def test_offers_only_the_four_cameras(self):
        schema = camera_panel_schema()
        assert [(c["type"], c["title"]) for c in schema] == [
            ("fixed", "Fixed view"),
            ("fly_around", "Fly to & around"),
            ("chase", "Follow a subject"),
            ("follow", "Follow the action"),
        ]

    @pytest.mark.parametrize("type_", ["fixed", "fly_around", "chase", "follow"])
    def test_every_camera_is_described_the_google_way(self, type_):
        names = {f["name"] for f in _camera(camera_panel_schema(), type_)["fields"]}
        assert {"tilt", "heading", "fov", "altitude_mode", "altitude"} <= names
        assert not names & {"zoom_offset", "pitch", "bearing", "zoom", "min_height_m"}

    def test_range_is_in_metres_and_optional_where_the_data_can_frame_it(self):
        schema = camera_panel_schema()
        assert _field(_camera(schema, "fixed"), "range_m")["default"] is None
        assert _field(_camera(schema, "chase"), "range_m")["default"] == 800

    def test_fov_is_limited_to_a_real_lens(self):
        fov = _field(_camera(camera_panel_schema(), "fixed"), "fov")
        assert (fov["min"], fov["max"], fov["default"]) == (5, 80, None)

    def test_altitude_modes_are_the_google_ones(self):
        mode = _field(_camera(camera_panel_schema(), "fly_around"), "altitude_mode")
        assert mode["type"] == "enum"
        assert mode["options"] == ["clamp_to_ground", "relative_to_ground", "absolute"]
        assert mode["default"] == "clamp_to_ground"

    def test_subject_lists_are_arrays(self):
        assert _field(_camera(camera_panel_schema(), "follow"), "subjects")["type"] == "array"

    def test_schema_is_json_serialisable(self):
        json.dumps(camera_panel_schema())


class TestParseCameraSettings:
    def test_pasted_text(self):
        camera = parse_camera_settings('{"type_": "fly_around", "range_m": 2500, "tilt": 55, "rounds": 2}')
        assert camera == FlyAroundCamera(range_m=2500, tilt=55, rounds=2)

    def test_downloaded_file(self, tmp_path):
        f = tmp_path / "camera-settings.json"
        f.write_text(json.dumps({"type_": "chase", "subject": "Amani", "range_m": 500, "heading": 90}))
        assert parse_camera_settings(str(f)) == FollowSubjectCamera(subject="Amani", range_m=500, heading=90)

    def test_file_uri(self, tmp_path):
        f = tmp_path / "camera-settings.json"
        f.write_text('{"type_": "fixed", "fov": 60}')
        assert parse_camera_settings(f.as_uri()) == FixedCamera(fov=60)

    def test_invalid_json_is_reported(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_camera_settings("{type_: fixed")

    @pytest.mark.parametrize(
        "settings",
        [
            '{"type_": "fixed", "tilt": 85}',
            '{"type_": "fixed", "fov": 2}',
            '{"type_": "chase", "range_m": 0}',
            '{"type_": "follow", "follow_window": 3}',
            '{"type_": "fly_around", "altitude_mode": "relative_to_mesh"}',
        ],
    )
    def test_out_of_range_values_are_rejected(self, settings):
        with pytest.raises(ValidationError):
            parse_camera_settings(settings)

    @pytest.mark.parametrize("type_", ["static", "orbit", "keyframes", "sequence", "tour", "pan"])
    def test_removed_cameras_are_rejected(self, type_):
        with pytest.raises(ValidationError):
            parse_camera_settings(json.dumps({"type_": type_}))


class TestCameraOptions:
    def test_fields_reach_the_page_with_the_video_size(self):
        camera = FollowActionCamera(subjects=["Amani"], tilt=50, heading=30, face_travel=True, fov=45)
        opts = _camera_opts(camera, 1080, 1920, 25)
        assert opts["preset"] == "follow" and "type_" not in opts
        assert (opts["tilt"], opts["heading"], opts["face_travel"], opts["fov"]) == (50, 30, True, 45)
        assert (opts["width"], opts["height"], opts["fps"]) == (1080, 1920, 25)

    def test_altitude_mode_reaches_the_page(self):
        opts = _camera_opts(FixedCamera(altitude_mode="absolute", altitude=1800), 1280, 720)
        assert (opts["altitude_mode"], opts["altitude"]) == ("absolute", 1800)

    def test_follow_subject_uses_the_chase_engine(self):
        opts = _camera_opts(FollowSubjectCamera(subject="Amani"), 1280, 720)
        assert (opts["preset"], opts["subject"], opts["range_m"], opts["tilt"]) == ("chase", "Amani", 800, 60)

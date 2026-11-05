"""Tests for the camera settings round trip between draw_animated_map's Camera panel and
render_animation: the panel's form schema and parsing the settings it copies or downloads."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ecoscope_workflows_ext_mep.tasks.animate._animate import (
    CameraKeyframe,
    CinematicCamera,
    SubjectTourCamera,
    FixedCamera,
    FlyAroundCamera,
    FollowActionCamera,
    FollowSubjectCamera,
    KeyframesCamera,
    CameraFromSettings,
    KeyframesFromFile,
    KeyframesFromSubject,
    _camera_opts,
    _overlay_css,
    _raw_keyframes,
    VideoOverlays,
    camera_panel_initial,
    camera_panel_schema,
    parse_camera_settings,
    set_video_camera,
)
from ecoscope_workflows_ext_mep.tasks.animate._draw_map import PlaybackControls, create_playback_controls


def _camera(schema, type_):
    return next(c for c in schema if c["type"] == type_)


def _field(camera, name):
    return next(f for f in camera["fields"] if f["name"] == name)


class TestCameraPanelSchema:
    def test_offers_only_the_seven_cameras(self):
        schema = camera_panel_schema()
        assert [(c["type"], c["title"]) for c in schema] == [
            ("fixed", "Fixed view"),
            ("fly_around", "Fly to & around"),
            ("chase", "Follow a subject"),
            ("tour", "Subject tour"),
            ("follow", "Follow the action"),
            ("cinematic", "Cinematic fly-through"),
            ("keyframes", "Keyframes"),
        ]

    def test_keyframes_waypoints_and_source_are_edited_by_the_panel_itself(self):
        names = {f["name"] for f in _camera(camera_panel_schema(), "keyframes")["fields"]}
        assert names == {"keyframe_easing", "fov"}

    @pytest.mark.parametrize("type_", ["fixed", "fly_around", "chase", "tour", "follow", "cinematic"])
    def test_every_moving_camera_is_described_the_google_way(self, type_):
        names = {f["name"] for f in _camera(camera_panel_schema(), type_)["fields"]}
        assert {"tilt", "heading", "fov"} <= names
        assert not names & {"zoom_offset", "pitch", "bearing", "zoom", "min_height_m", "altitude_mode", "altitude"}

    def test_range_is_in_metres_and_optional_where_the_data_can_frame_it(self):
        schema = camera_panel_schema()
        assert _field(_camera(schema, "fixed"), "range_m")["default"] is None
        assert _field(_camera(schema, "chase"), "range_m")["default"] == 800

    def test_fov_is_limited_to_a_real_lens(self):
        fov = _field(_camera(camera_panel_schema(), "fixed"), "fov")
        assert (fov["min"], fov["max"], fov["default"]) == (5, 80, None)

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
        ],
    )
    def test_out_of_range_values_are_rejected(self, settings):
        with pytest.raises(ValidationError):
            parse_camera_settings(settings)

    @pytest.mark.parametrize("type_", ["static", "orbit", "sequence", "pan"])
    def test_removed_cameras_are_rejected(self, type_):
        with pytest.raises(ValidationError):
            parse_camera_settings(json.dumps({"type_": type_}))


class TestSetVideoCamera:
    """One camera for the map's Camera panel and the video, so the preview matches the export."""

    # What Jake pasted: the map kept playing in Fixed view while the video flew around.
    PASTED = '{"type_":"fly_around","fly_frac":0.25,"rounds":1}'

    def test_pasted_settings_become_the_camera(self):
        camera = set_video_camera(camera_settings=CameraFromSettings(settings=self.PASTED))
        assert camera == FlyAroundCamera(fly_frac=0.25, rounds=1)

    def test_a_camera_passes_through(self):
        assert set_video_camera(camera_settings=FixedCamera(tilt=30)) == FixedCamera(tilt=30)

    def test_default_is_a_fixed_view(self):
        assert set_video_camera() == FixedCamera()

    def test_a_bad_waypoint_file_fails_before_rendering(self):
        with pytest.raises(ValueError, match="keyframes_file"):
            set_video_camera(camera_settings=KeyframesCamera(source=KeyframesFromFile()))

    def test_the_panel_opens_on_the_pasted_camera(self):
        camera = set_video_camera(camera_settings=CameraFromSettings(settings=self.PASTED))
        assert camera_panel_initial(camera) == {"type_": "fly_around", "fly_frac": 0.25, "rounds": 1.0}


class TestSubjectTourCamera:
    """Each subject in turn, flying between them, with name cards."""

    def test_pasted_settings_become_the_camera(self):
        camera = set_video_camera(
            camera_settings=CameraFromSettings(settings='{"type_": "tour", "subjects": ["Kito", "Sopa"]}')
        )
        assert camera == SubjectTourCamera(subjects=["Kito", "Sopa"])

    def test_reaches_the_page_as_the_tour_preset(self):
        opts = _camera_opts(SubjectTourCamera(name_cards=False, transition_frac=0.3), 1280, 720, 25)
        assert (opts["preset"], opts["name_cards"], opts["transition_frac"], opts["fps"]) == ("tour", False, 0.3, 25)
        assert (opts["range_m"], opts["tilt"], opts["lead_room"], opts["speed_zoom"]) == (800, 60, 0.15, 0.5)

    def test_settings_saved_before_it_was_removed_still_parse(self):
        # The earlier Subject tour carried zoom_offset, fit_padding and min_height_m.
        camera = parse_camera_settings(
            '{"type_": "tour", "subjects": ["Kito"], "transition_frac": 0.6, "zoom_offset": 1, "fit_padding": 40}'
        )
        assert camera == SubjectTourCamera(subjects=["Kito"], transition_frac=0.6)

    def test_panel_offers_its_settings(self):
        names = {f["name"] for f in _camera(camera_panel_schema(), "tour")["fields"]}
        assert {"subjects", "name_cards", "transition_frac", "lead_room", "speed_zoom", "keep_in_view"} <= names


class TestFollowingCameraOptions:
    """Lead room, zoom with the action and keeping the subject in view over hills."""

    @pytest.mark.parametrize(
        "camera, lead, zoom",
        [
            (FollowSubjectCamera(), 0.15, 0.5),
            (SubjectTourCamera(), 0.15, 0.5),
            (FollowActionCamera(), 0.1, 0.3),
            (CinematicCamera(), 0.1, 0.3),
        ],
    )
    def test_defaults(self, camera, lead, zoom):
        assert (camera.lead_room, camera.speed_zoom, camera.keep_in_view) == (lead, zoom, True)

    def test_they_reach_the_page(self):
        opts = _camera_opts(FollowSubjectCamera(lead_room=0, speed_zoom=1, keep_in_view=False), 1280, 720)
        assert (opts["lead_room"], opts["speed_zoom"], opts["keep_in_view"]) == (0, 1, False)

    @pytest.mark.parametrize("field, bad", [("lead_room", 0.6), ("speed_zoom", 1.5), ("speed_zoom", -0.1)])
    def test_limits(self, field, bad):
        with pytest.raises(ValidationError):
            FollowSubjectCamera(**{field: bad})

    @pytest.mark.parametrize("type_", ["chase", "tour", "follow", "cinematic"])
    def test_panel_offers_them_on_every_following_camera(self, type_):
        names = {f["name"] for f in _camera(camera_panel_schema(), type_)["fields"]}
        assert {"lead_room", "speed_zoom", "keep_in_view"} <= names


class TestCinematicCamera:
    """Cinematic fly-through, back as a camera choice (Jake's request)."""

    def test_pasted_settings_become_the_camera(self):
        camera = set_video_camera(
            camera_settings=CameraFromSettings(settings='{"type_": "cinematic", "rotate_deg": 90, "intro_frac": 0.2}')
        )
        assert camera == CinematicCamera(rotate_deg=90, intro_frac=0.2)

    def test_reaches_the_page_as_the_cinematic_preset(self):
        opts = _camera_opts(CinematicCamera(bearing_mode="heading", subjects=["Kito"]), 1280, 720)
        assert (opts["preset"], opts["bearing_mode"], opts["subjects"]) == ("cinematic", "heading", ["Kito"])
        assert (opts["tilt"], opts["rotate_deg"], opts["intro_frac"]) == (55, 45, 0.12)

    def test_settings_saved_before_it_was_removed_still_parse(self):
        # Pre-Google-style cinematic settings carried zoom_offset and min_height_m.
        camera = parse_camera_settings('{"type_": "cinematic", "zoom_offset": 1, "min_height_m": 200}')
        assert camera == CinematicCamera()

    def test_panel_offers_the_turning_choices(self):
        field = _field(_camera(camera_panel_schema(), "cinematic"), "bearing_mode")
        assert (field["type"], field["options"], field["default"]) == ("enum", ["rotate", "heading", "fixed"], "rotate")

    def test_the_panel_opens_on_it(self):
        assert camera_panel_initial(CinematicCamera(rotate_deg=-30)) == {
            **CinematicCamera().model_dump(mode="json", exclude_none=True),
            "rotate_deg": -30.0,
        }


class TestCameraPanelInitial:
    def test_none_leaves_the_panel_on_its_first_camera(self):
        assert camera_panel_initial(None) is None

    def test_blank_values_are_left_out(self):
        assert camera_panel_initial(FixedCamera(range_m=900)) == {"type_": "fixed", "range_m": 900}

    def test_panel_settings_round_trip(self):
        camera = FollowSubjectCamera(subject="Amani", range_m=500, heading=90)
        assert parse_camera_settings(json.dumps(camera_panel_initial(camera))) == camera

    def test_waypoints_come_sorted_with_times(self):
        camera = KeyframesCamera(keyframes=[CameraKeyframe(lon=1, lat=1, zoom=9), CameraKeyframe(lon=0, lat=0)])
        out = camera_panel_initial(camera)
        assert "source" not in out
        assert [(k["t"], k["lon"], k["zoom"]) for k in out["keyframes"]] == [(0, 1, 9), (1, 0, 9)]

    def test_a_waypoint_file_is_read_in(self, tmp_path):
        f = tmp_path / "path.csv"
        f.write_text("lon,lat\n37.0,-1.0\n37.1,-1.1\n")
        out = camera_panel_initial(KeyframesCamera(source=KeyframesFromFile(keyframes_file=str(f))))
        assert [(k["lon"], k["lat"]) for k in out["keyframes"]] == [(37.0, -1.0), (37.1, -1.1)]
        assert "source" not in out

    def test_no_waypoints_keeps_the_subject_to_follow(self):
        out = camera_panel_initial(KeyframesCamera(source=KeyframesFromSubject(subject="Amani")))
        assert out["source"] == {"type_": "subject", "subject": "Amani"} and "keyframes" not in out


class TestPlaybackControlDefaults:
    def test_preview_and_hint_are_on_by_default(self):
        assert PlaybackControls().preview_camera and PlaybackControls().show_hint
        controls = create_playback_controls()
        assert controls.preview_camera and controls.show_hint

    def test_the_video_hides_the_hint_and_the_preview_note(self):
        css = _overlay_css(VideoOverlays(), 720)
        assert "#map-hint" in css and "#camera-toast" in css


class TestCameraOptions:
    def test_fields_reach_the_page_with_the_video_size(self):
        camera = FollowActionCamera(subjects=["Amani"], tilt=50, heading=30, face_travel=True, fov=45)
        opts = _camera_opts(camera, 1080, 1920, 25)
        assert opts["preset"] == "follow" and "type_" not in opts
        assert (opts["tilt"], opts["heading"], opts["face_travel"], opts["fov"]) == (50, 30, True, 45)
        assert (opts["width"], opts["height"], opts["fps"]) == (1080, 1920, 25)

    def test_settings_saved_with_an_altitude_mode_still_parse(self):
        # Camera panel files saved before altitude_mode/altitude were removed.
        camera = parse_camera_settings('{"type_": "fixed", "altitude_mode": "absolute", "altitude": 1800}')
        assert camera == FixedCamera()
        assert "altitude_mode" not in _camera_opts(camera, 1280, 720)

    def test_follow_subject_uses_the_chase_engine(self):
        opts = _camera_opts(FollowSubjectCamera(subject="Amani"), 1280, 720)
        assert (opts["preset"], opts["subject"], opts["range_m"], opts["tilt"]) == ("chase", "Amani", 800, 60)


class TestKeyframesCamera:
    # What the Camera panel copies after two "+ Add current view" clicks.
    PANEL = (
        '{"type_": "keyframes", "keyframe_easing": "spline", "keyframes": ['
        '{"t": 0.8, "lon": 37.2, "lat": -1.1, "zoom": 12, "pitch": 50, "bearing": 170},'
        '{"t": 0.2, "lon": 37.0, "lat": -1.0, "zoom": 10, "pitch": 30, "bearing": -170}]}'
    )

    def test_panel_waypoints_reach_the_page_sorted_and_normalised(self):
        opts = _camera_opts(parse_camera_settings(self.PANEL), 1280, 720)
        assert opts["preset"] == "keyframes" and opts["keyframe_easing"] == "spline"
        assert "source" not in opts
        first, last = opts["keyframes"]
        assert (first["t"], first["lon"], last["t"], last["lon"]) == (0.0, 37.0, 1.0, 37.2)
        assert last["bearing"] == -190  # -170 -> 170 turns 20° the short way, not 340°

    def test_unset_channels_blend_between_waypoints_or_are_left_to_the_page(self):
        camera = KeyframesCamera(
            keyframes=[
                CameraKeyframe(lon=0, lat=0, zoom=8),
                CameraKeyframe(lon=1, lat=1),
                CameraKeyframe(lon=2, lat=2, zoom=12),
            ]
        )
        k = _camera_opts(camera, 1280, 720)["keyframes"]
        assert [x["t"] for x in k] == [0, 0.5, 1]
        assert [x["zoom"] for x in k] == [8, 10, 12]
        assert all(x["pitch"] is None and x["bearing"] is None for x in k)  # the map's tilt and heading

    def test_no_waypoints_follows_the_source_subject(self):
        camera = parse_camera_settings('{"type_": "keyframes", "source": {"type_": "subject", "subject": "Amani"}}')
        opts = _camera_opts(camera, 1280, 720)
        assert opts["keyframes"] is None and opts["subject"] == "Amani"

    def test_one_waypoint_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2 keyframes"):
            _raw_keyframes(KeyframesCamera(keyframes=[CameraKeyframe(lon=0, lat=0)]))

    def test_file_source_needs_a_file(self):
        with pytest.raises(ValueError, match="keyframes_file"):
            _raw_keyframes(KeyframesCamera(source=KeyframesFromFile()))

    def test_csv_waypoint_file(self, tmp_path):
        f = tmp_path / "path.csv"
        f.write_text("longitude,latitude,heading\n37.0,-1.0,90\n37.1,-1.1,\n")
        camera = KeyframesCamera(source=KeyframesFromFile(keyframes_file=f.as_uri()))
        assert _raw_keyframes(camera) == [{"lon": 37.0, "lat": -1.0, "bearing": 90.0}, {"lon": 37.1, "lat": -1.1}]

    def test_geojson_waypoint_file(self, tmp_path):
        f = tmp_path / "path.geojson"
        points = [([37.0, -1.0], {"zoom": 11, "t": 0}), ([37.1, -1.1], {"Pitch": 45, "t": 1})]
        f.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": {"type": "Point", "coordinates": c}, "properties": p}
                        for c, p in points
                    ]
                    + [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}],
                }
            )
        )
        raw = _raw_keyframes(KeyframesCamera(source=KeyframesFromFile(keyframes_file=str(f))))
        assert raw == [
            {"lon": 37.0, "lat": -1.0, "t": 0.0, "zoom": 11.0},
            {"lon": 37.1, "lat": -1.1, "t": 1.0, "pitch": 45.0},
        ]

    def test_unsupported_file_type(self, tmp_path):
        f = tmp_path / "path.kml"
        f.write_text("<kml/>")
        with pytest.raises(ValueError, match="Unsupported keyframe file type"):
            _raw_keyframes(KeyframesCamera(source=KeyframesFromFile(keyframes_file=str(f))))

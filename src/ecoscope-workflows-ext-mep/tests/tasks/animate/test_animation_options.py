"""Tests for the trips animation's motion and look options and the timeline's sun lighting:
the task form passes each option through, and draw_animated_map hands them to the page script."""

from __future__ import annotations

import inspect
import json
import logging

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from ecoscope_workflows_ext_mep.tasks.animate._animate import camera_panel_schema
from ecoscope_workflows_ext_mep.tasks.animate._draw_map import (
    AnimalModelMarker,
    PydeckLayerDefinition,
    TimelineAnimation,
    TripsAnimation,
    TripsLayerStyle,
    _head_spec,
    animate_layer,
    create_playback_controls,
    create_scenegraph_layer,
    create_timeline_animation,
    create_trips_animation,
    draw_animated_map,
)


@pytest.fixture
def trips_layer():
    t0 = 1_790_000_000.0
    gdf = gpd.GeoDataFrame(
        [
            {
                "subject_name": "Luna",
                "rgba_color": [200, 180, 40, 255],
                "timestamps": [t0, t0 + 900, t0 + 1800, t0 + 60_000],  # last step: a 16.5 h gap
                "geometry": LineString([(38.0, 0.2), (38.01, 0.21), (38.02, 0.2), (38.1, 0.25)]),
            },
            {
                "subject_name": "Soutine",
                "rgba_color": [230, 100, 230, 255],
                "timestamps": [t0 + 100, t0 + 1000, t0 + 2000],
                "geometry": LineString([(37.6, 0.5), (37.61, 0.5), (37.62, 0.51)]),
            },
        ],
        crs="EPSG:4326",
    )
    style = TripsLayerStyle(get_timestamps="timestamps", get_color="rgba_color")
    return PydeckLayerDefinition(layer_type="TripsLayer", layer_style=style, legend=None, geodataframe=gdf)


def _widget_props(html: str) -> dict:
    """The TimelineAnimationWidget's props, from the pydeck spec embedded in the page."""
    start = html.index("window.jsonInput = ") + len("window.jsonInput = ")
    spec, _ = json.JSONDecoder().raw_decode(html, start)
    return next(w for w in spec["widgets"] if w["@@type"] == "TimelineAnimationWidget")


def _page_settings(html: str) -> tuple[dict, dict]:
    """The trips layer spec and the sun settings the widget was given."""
    props = _widget_props(html)
    return props["layerSpecs"][0], props["sun"]


def _draw(layer, animation, **timeline):
    # The model, not create_timeline_animation: most timeline options aren't task parameters.
    return draw_animated_map(
        geo_layers=animate_layer(layer=layer, animation=animation),
        timeline=TimelineAnimation(**timeline),
    )


class TestDefaults:
    def test_smooth_turning_is_on_and_curves_are_opt_in(self):
        anim = TripsAnimation()
        assert anim.smooth_turning and not anim.curve_paths and anim.turn_smoothing == 3.0
        assert anim.hover_info and anim.label_column == ""

    @pytest.mark.parametrize(
        "field",
        [
            "show_gaps",
            "gap_hours",
            "speed_response",
            "comet_color",
            "history_lag",
            "shadow",
            "glow",
            "glow_width",
            "glow_opacity",
            "pulse",
            "pulse_period_s",
            "click_to_follow",
        ],
    )
    def test_removed_options_are_gone(self, field):
        assert field not in TripsAnimation.model_fields
        assert field not in inspect.signature(create_trips_animation).parameters

    def test_timeline_has_sun_lighting_and_a_60_fps_cap(self):
        timeline = TimelineAnimation()
        assert timeline.sun_lighting and timeline.night_brightness == 0.45
        assert timeline.day_night_smoothing_s == 0.3
        assert timeline.fps_limit == 60.0

    def test_task_defaults_match_the_models(self):
        assert create_trips_animation() == TripsAnimation()
        assert create_timeline_animation() == TimelineAnimation()


class TestTasksPassOptionsThrough:
    def test_trips_options(self):
        anim = create_trips_animation(
            smooth_turning=False,
            turn_smoothing=5,
            curve_paths=True,
            label_column="subject_name",
            hover_info=False,
        )
        assert anim.model_dump(include={"smooth_turning", "turn_smoothing", "curve_paths"}) == {
            "smooth_turning": False,
            "turn_smoothing": 5,
            "curve_paths": True,
        }
        assert anim.label_column == "subject_name" and not anim.hover_info

    def test_timeline_task_options(self):
        timeline = create_timeline_animation(duration=12, fps_limit=30)
        assert (timeline.duration_s, timeline.fps_limit) == (12, 30)

    def test_timeline_task_leaves_feel_options_at_the_model_defaults(self):
        hidden = {
            "duration_s",
            "auto_rotate_speed",
            "sun_lighting",
            "night_brightness",
            "day_night_smoothing_s",
            "adaptive_speed",
            "max_speedup",
            "rest_speed_kmh",
            "fast_forward_night",
            "ease_s",
            "follow_smoothing_s",
        }
        assert hidden.isdisjoint(inspect.signature(create_timeline_animation).parameters)
        timeline = create_timeline_animation()
        hidden.discard("duration_s")  # renamed to `duration` on the task, same default
        assert {k: getattr(timeline, k) for k in hidden} == {
            k: TimelineAnimation.model_fields[k].default for k in hidden
        }

    def test_model_animations(self):
        model = create_scenegraph_layer(walk_animation="walk", idle_animation="Idle1", animation_speed=1.5)
        anims = _head_spec(model)["model"]["anims"]
        assert anims == {"walk": "walk", "run": "walk", "idle": "Idle1", "speed": 1.5}  # run falls back to walk

    @pytest.mark.parametrize(
        "animal,expected",
        [("elephant", ("12", "11", "7")), ("lion", ("0", "0", None)), ("giraffe", (None, None, None))],
    )
    def test_preset_animals_name_their_clips(self, animal, expected):
        anims = _head_spec(AnimalModelMarker(animal=animal))["model"]["anims"]
        assert (anims["walk"], anims["run"], anims["idle"]) == expected

    @pytest.mark.parametrize(
        "field,value",
        [("turn_smoothing", 0), ("comet_ratio", 1.5), ("night_brightness", 1.5), ("max_speedup", 1)],
    )
    def test_out_of_range_values_are_rejected(self, field, value):
        # The task if the field is one of its parameters; else the model, which still validates it.
        if field in TimelineAnimation.model_fields:
            task, model = create_timeline_animation, TimelineAnimation
        else:
            task, model = create_trips_animation, TripsAnimation
        build = task if field in inspect.signature(task).parameters else model
        with pytest.raises(ValueError):
            build(**{field: value})


class TestPageSettings:
    def test_defaults_reach_the_page(self, trips_layer):
        spec, sun = _page_settings(_draw(trips_layer, TripsAnimation()))
        assert spec["smoothTurning"] is True and spec["turnSmoothing"] == 3.0
        assert spec["curvePaths"] is False
        assert spec["labelCol"] == ""
        assert spec["hoverInfo"] is True and spec["speedResponse"] is True
        assert sun == {"on": True, "night": 0.45, "smoothS": 0.3}

    def test_removed_effects_are_off_on_the_page(self, trips_layer):
        # The widget treats a missing key as off: no glow, pulse, shadow, gap hours, history lag,
        # speed / day-night comet colours or click-to-follow.
        spec, _ = _page_settings(_draw(trips_layer, TripsAnimation()))
        removed = {"gapS", "glow", "pulsePeriod", "clickToFollow", "cometColor", "historyLag", "shadow"}
        assert removed.isdisjoint(spec)

    def test_feel_defaults_reach_the_page(self, trips_layer):
        html = _draw(trips_layer, TripsAnimation())
        props = _widget_props(html)
        assert props["pacing"] == {"adaptive": False, "maxSpeedup": 6.0, "restKmh": 0.5, "night": False}
        assert (props["easeS"], props["followSmoothingS"]) == (0.6, 0.4)
        # Exaggeration comes from create_elevation_decoder (terrain and sampled heights), not the widget.
        assert "terrainExaggeration" not in props

    def test_adaptive_speed_reaches_the_page(self, trips_layer):
        props = _widget_props(_draw(trips_layer, TripsAnimation(), adaptive_speed=True, max_speedup=10))
        assert props["pacing"]["adaptive"] is True and props["pacing"]["maxSpeedup"] == 10

    def test_sun_lighting_can_be_switched_off(self, trips_layer):
        _, sun = _page_settings(_draw(trips_layer, TripsAnimation(), sun_lighting=False))
        assert sun["on"] is False

    def test_label_column_is_used_for_labels_and_hover(self, trips_layer):
        spec, _ = _page_settings(_draw(trips_layer, TripsAnimation(label_column="subject_name")))
        assert spec["labelCol"] == "subject_name" and spec["tipCol"] == "subject_name"

    def test_unknown_label_column_is_dropped_with_a_warning(self, trips_layer, caplog):
        with caplog.at_level(logging.WARNING):
            spec, _ = _page_settings(_draw(trips_layer, TripsAnimation(label_column="nickname")))
        assert spec["labelCol"] == "" and spec["tipCol"] == ""
        assert "nickname" in caplog.text

    def test_timeline_reaches_the_widget(self, trips_layer):
        props = _widget_props(_draw(trips_layer, TripsAnimation(), duration_s=12, fps_limit=30))
        assert (props["durationS"], props["fpsLimit"], props["autoRotateSpeed"]) == (12, 30, 0)
        assert props["t0"] == 1_790_000_000.0 and props["maxTime"] == pytest.approx(60_000 * 1.02)

    def test_map_opens_fitted_to_the_tracks(self, trips_layer):
        props = _widget_props(_draw(trips_layer, TripsAnimation()))
        assert props["fitView"] == {"padding": 80, "maxZoom": 20}

    def test_fit_view_off_keeps_the_view_state(self, trips_layer):
        html = draw_animated_map(
            geo_layers=animate_layer(layer=trips_layer, animation=TripsAnimation()), fit_view=False
        )
        assert _widget_props(html).get("fitView") is None  # pydeck drops None props

    def test_camera_panel_widget_gets_the_camera_schema(self, trips_layer):
        html = _draw(trips_layer, TripsAnimation())
        spec = json.JSONDecoder().raw_decode(html, html.index("window.jsonInput = ") + len("window.jsonInput = "))[0]
        panel = next(w for w in spec["widgets"] if w["@@type"] == "CameraPanelWidget")
        assert panel["cameras"] == camera_panel_schema()
        assert "window.jsonInput = " in html  # read by the bundle's video camera (window.__cam)

    def test_no_camera_panel_without_the_camera_button(self, trips_layer):
        html = _draw(trips_layer, TripsAnimation(), controls=create_playback_controls(show_camera=False))
        assert "CameraPanelWidget" not in html

    def test_widget_bundle_is_registered(self, trips_layer):
        html = _draw(trips_layer, TripsAnimation())
        assert "my-deck-widgets" in html and "<script>\n// === Timeline animation" not in html


def _panel_props(html: str) -> dict:
    start = html.index("window.jsonInput = ") + len("window.jsonInput = ")
    spec, _ = json.JSONDecoder().raw_decode(html, start)
    return next(w for w in spec["widgets"] if w["@@type"] == "CameraPanelWidget")


class TestCameraPanelGetsTheWorkflowCamera:
    """The map's Camera panel opens on, and previews, the camera the video uses."""

    def test_panel_opens_on_the_camera_with_the_preview_on(self, trips_layer):
        from ecoscope_workflows_ext_mep.tasks.animate._animate import FlyAroundCamera

        html = draw_animated_map(
            geo_layers=animate_layer(layer=trips_layer, animation=TripsAnimation()),
            camera=FlyAroundCamera(fly_frac=0.25, rounds=1),
        )
        props = _panel_props(html)
        assert props["initial"] == {"type_": "fly_around", "fly_frac": 0.25, "rounds": 1.0}
        assert props["preview"] is True
        assert _widget_props(html)["controls"]["show_hint"] is True

    def test_no_camera_and_preview_off(self, trips_layer):
        html = _draw(trips_layer, TripsAnimation(), controls=create_playback_controls(preview_camera=False))
        props = _panel_props(html)
        assert props.get("initial") is None and props["preview"] is False


class TestTerrainHasNoWhiteGaps:
    """Over a 3D terrain the page is dark, not a light base map that shows through as white gaps."""

    def _layer_json(self, html: str) -> dict:
        start = html.index("window.jsonInput = ") + len("window.jsonInput = ")
        spec, _ = json.JSONDecoder().raw_decode(html, start)
        return spec, next(layer for layer in spec["layers"] if layer["@@type"] == "TerrainLayer")

    def test_terrain_page_is_dark_with_no_base_style(self, trips_layer):
        from ecoscope_workflows_ext_mep.tasks.animate._draw_map import TERRAIN_BACKGROUND, create_terrain_layer

        html = draw_animated_map(
            geo_layers=animate_layer(layer=trips_layer, animation=TripsAnimation()),
            tile_layers=[create_terrain_layer()],
        )
        spec, terrain = self._layer_json(html)
        assert not spec.get("mapStyle")
        assert TERRAIN_BACKGROUND in html
        # deck.gl's prop names: the old `strategy` was ignored; color replaces the white of unloaded tiles.
        assert terrain["refinementStrategy"] == "best-available" and "strategy" not in terrain
        assert terrain["color"] == [58, 56, 48]

    def test_without_terrain_the_light_base_map_stays(self, trips_layer):
        html = draw_animated_map(geo_layers=animate_layer(layer=trips_layer, animation=TripsAnimation()))
        start = html.index("window.jsonInput = ") + len("window.jsonInput = ")
        spec, _ = json.JSONDecoder().raw_decode(html, start)
        assert spec["mapStyle"]


class TestDeckGlMapControls:
    """deck.gl's own widgets (@deck.gl/widgets) for zoom, tilt/rotate and reset, in its widget theme."""

    def _widget_types(self, html: str) -> list[str]:
        start = html.index("window.jsonInput = ") + len("window.jsonInput = ")
        spec, _ = json.JSONDecoder().raw_decode(html, start)
        return [w["@@type"] for w in spec["widgets"]]

    def test_controls_are_on_by_default_and_before_the_camera_panel(self, trips_layer):
        html = draw_animated_map(geo_layers=animate_layer(layer=trips_layer, animation=TripsAnimation()))
        types = self._widget_types(html)
        for w in ("ZoomWidget", "CompassWidget", "ResetViewWidget", "FullscreenWidget", "LoadingWidget"):
            assert w in types
        assert types.index("FullscreenWidget") < types.index("CameraPanelWidget")
        assert "--button-background" in html  # the dark widget theme

    def test_controls_can_be_switched_off(self, trips_layer):
        html = _draw(trips_layer, TripsAnimation(), controls=create_playback_controls(show_map_controls=False))
        assert "ZoomWidget" not in self._widget_types(html)

    def test_the_video_hides_them(self):
        from ecoscope_workflows_ext_mep.tasks.animate._animate import VideoOverlays, _overlay_css

        css = _overlay_css(VideoOverlays(), 720)
        for cls in ("zoom", "compass", "reset-view", "fullscreen", "loading"):
            assert f".deck-widget-{cls}" in css


class TestDrapedBasemap:
    """Basemap tiles draped over the terrain at their own zoom (my-deck-widgets' DrapedTileLayer)."""

    def _spec(self, trips_layer, **terrain_kw):
        from ecoscope_workflows_ext_mep.tasks.animate._draw_map import create_terrain_layer

        html = draw_animated_map(
            geo_layers=animate_layer(layer=trips_layer, animation=TripsAnimation()),
            tile_layers=[create_terrain_layer(**terrain_kw)],
        )
        start = html.index("window.jsonInput = ") + len("window.jsonInput = ")
        spec, _ = json.JSONDecoder().raw_decode(html, start)
        terrain = next(layer for layer in spec["layers"] if layer["@@type"] == "TerrainLayer")
        draped = {layer["id"]: layer for layer in spec["layers"] if layer["@@type"] == "DrapedTileLayer"}
        return spec, terrain, draped

    def test_imagery_is_draped_to_zoom_18_by_default(self, trips_layer):
        spec, terrain, draped = self._spec(trips_layer)
        assert draped["draped-imagery"]["maxZoom"] == 18
        assert draped["draped-imagery"]["data"] == terrain["texture"]
        assert terrain["operation"] == "terrain+draw"  # a literal, not an @@= expression
        assert "draped-labels" not in draped
        assert "imageryMaxZoom" not in terrain and "labels" not in terrain  # not TerrainLayer props
        # Depth test on over terrain, so tiles don't paint over each other (the cracks between tiles).
        assert spec["parameters"]["depthTest"] is True

    def test_place_names(self, trips_layer):
        _, _, draped = self._spec(trips_layer, labels=True)
        assert "World_Boundaries_and_Places" in draped["draped-labels"]["data"]

    def test_terrain_zoom_only(self, trips_layer):
        _, terrain, draped = self._spec(trips_layer, imagery_max_zoom=None)
        assert not draped and "operation" not in terrain

    def test_no_drape_when_not_deeper_than_the_terrain(self, trips_layer):
        _, _, draped = self._spec(trips_layer, imagery_max_zoom=15)
        assert not draped


class TestCompassReplacesNorthArrow:
    def test_no_gimbal_and_the_static_north_arrow_hides_behind_the_compass(self, trips_layer):
        html = draw_animated_map(geo_layers=animate_layer(layer=trips_layer, animation=TripsAnimation()))
        assert "GimbalWidget" not in html  # OrbitView-only: does nothing on a map
        assert ".ecoscope-north-arrow-widget { display: none; }" in html

    def test_north_arrow_stays_without_map_controls(self, trips_layer):
        html = _draw(trips_layer, TripsAnimation(), controls=create_playback_controls(show_map_controls=False))
        assert ".ecoscope-north-arrow-widget { display: none; }" not in html

    def test_the_video_brings_the_north_arrow_back(self):
        from ecoscope_workflows_ext_mep.tasks.animate._animate import VideoOverlays, _overlay_css

        assert ".ecoscope-north-arrow-widget { display: block !important; }" in _overlay_css(VideoOverlays(), 720)
        off = _overlay_css(VideoOverlays(north_arrow=False), 720)
        assert "display: block" not in off and ".ecoscope-north-arrow-widget" in off

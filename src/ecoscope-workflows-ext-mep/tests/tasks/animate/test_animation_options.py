"""Tests for the trips animation's motion and look options and the timeline's sun lighting:
the task form passes each option through, and draw_animated_map hands them to the page script."""

from __future__ import annotations

import json
import logging
import re

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from ecoscope_workflows_ext_mep.tasks.animate._draw_map import (
    LayerDefinition,
    TimelineAnimation,
    TripsAnimation,
    TripsLayerStyle,
    animate_layer,
    create_playback_controls,
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
    return LayerDefinition(layer_type="TripsLayer", layer_style=style, legend=None, geodataframe=gdf)


def _page_settings(html: str) -> tuple[dict, dict]:
    """The trips layer spec and the sun settings the page script was given."""
    specs = json.loads(re.search(r"const LAYER_SPECS\s+= (\[.*?\]);\n", html).group(1))
    sun = json.loads(re.search(r"const SUN\s+= (\{.*?\});", html).group(1))
    return specs[0], sun


def _draw(layer, animation, **timeline):
    return draw_animated_map(
        geo_layers=animate_layer(layer=layer, animation=animation),
        timeline=create_timeline_animation(**timeline),
    )


class TestDefaults:
    def test_fixes_are_on_and_curves_are_opt_in(self):
        anim = TripsAnimation()
        assert anim.smooth_turning and anim.show_gaps and not anim.curve_paths
        assert (anim.turn_smoothing, anim.gap_hours) == (3.0, 12.0)
        assert anim.glow and anim.pulse and anim.hover_info and anim.click_to_follow
        assert anim.label_column == ""

    def test_timeline_has_sun_lighting_and_a_60_fps_cap(self):
        timeline = TimelineAnimation()
        assert timeline.sun_lighting and timeline.night_brightness == 0.45
        assert timeline.day_night_smoothing_s == 0.3
        assert timeline.fps_limit == 60.0
        assert timeline.controls.utc_offset_hours == 0

    def test_task_defaults_match_the_models(self):
        assert create_trips_animation() == TripsAnimation()
        assert create_timeline_animation() == TimelineAnimation()


class TestTasksPassOptionsThrough:
    def test_trips_options(self):
        anim = create_trips_animation(
            smooth_turning=False,
            turn_smoothing=5,
            curve_paths=True,
            show_gaps=False,
            gap_hours=3,
            glow=False,
            glow_width=4,
            glow_opacity=0.5,
            pulse=False,
            pulse_period_s=2,
            label_column="subject_name",
            hover_info=False,
            click_to_follow=False,
        )
        assert anim.model_dump(include={"smooth_turning", "turn_smoothing", "curve_paths", "show_gaps"}) == {
            "smooth_turning": False,
            "turn_smoothing": 5,
            "curve_paths": True,
            "show_gaps": False,
        }
        assert (anim.gap_hours, anim.glow, anim.glow_width, anim.glow_opacity) == (3, False, 4, 0.5)
        assert (anim.pulse, anim.pulse_period_s, anim.label_column) == (False, 2, "subject_name")
        assert not anim.hover_info and not anim.click_to_follow

    def test_timeline_options(self):
        timeline = create_timeline_animation(sun_lighting=False, night_brightness=0.2, day_night_smoothing_s=1)
        assert not timeline.sun_lighting and timeline.night_brightness == 0.2
        assert timeline.day_night_smoothing_s == 1

    def test_clock_utc_offset(self):
        assert create_playback_controls(utc_offset_hours=5.5).utc_offset_hours == 5.5
        with pytest.raises(ValueError):
            create_playback_controls(utc_offset_hours=15)

    @pytest.mark.parametrize("field,value", [("glow_width", 1), ("turn_smoothing", 0), ("night_brightness", 1.5)])
    def test_out_of_range_values_are_rejected(self, field, value):
        build = create_timeline_animation if field == "night_brightness" else create_trips_animation
        with pytest.raises(ValueError):
            build(**{field: value})


class TestPageSettings:
    def test_defaults_reach_the_page(self, trips_layer):
        spec, sun = _page_settings(_draw(trips_layer, TripsAnimation()))
        assert spec["smoothTurning"] is True and spec["turnSmoothing"] == 3.0
        assert spec["curvePaths"] is False
        assert spec["gapS"] == 12 * 3600
        assert spec["glow"] == {"width": 3.0, "opacity": 0.25}
        assert spec["pulsePeriod"] == 1.5
        assert spec["labelCol"] == ""
        assert spec["hoverInfo"] is True and spec["clickToFollow"] is True
        assert sun == {"on": True, "night": 0.45, "smoothS": 0.3}

    def test_switched_off_options_are_null(self, trips_layer):
        anim = TripsAnimation(show_gaps=False, glow=False, pulse=False)
        spec, sun = _page_settings(_draw(trips_layer, anim, sun_lighting=False))
        assert spec["gapS"] is None and spec["glow"] is None and spec["pulsePeriod"] is None
        assert sun["on"] is False

    def test_clock_offset_reaches_the_page(self, trips_layer):
        controls = create_playback_controls(utc_offset_hours=3)
        html = _draw(trips_layer, TripsAnimation(), controls=controls)
        page_controls = json.loads(re.search(r"const CONTROLS\s+= (\{.*?\});", html).group(1))
        assert page_controls["utc_offset_hours"] == 3

    def test_gap_hours_become_seconds(self, trips_layer):
        spec, _ = _page_settings(_draw(trips_layer, TripsAnimation(gap_hours=2.5)))
        assert spec["gapS"] == 9000

    def test_label_column_is_used_for_labels_and_hover(self, trips_layer):
        spec, _ = _page_settings(_draw(trips_layer, TripsAnimation(label_column="subject_name")))
        assert spec["labelCol"] == "subject_name" and spec["tipCol"] == "subject_name"

    def test_unknown_label_column_is_dropped_with_a_warning(self, trips_layer, caplog):
        with caplog.at_level(logging.WARNING):
            spec, _ = _page_settings(_draw(trips_layer, TripsAnimation(label_column="nickname")))
        assert spec["labelCol"] == "" and spec["tipCol"] == ""
        assert "nickname" in caplog.text

    def test_no_placeholder_is_left_unfilled(self, trips_layer):
        html = _draw(trips_layer, TripsAnimation())
        assert not re.search(r"__[A-Z_]+__", html.split("const LAYER_SPECS", 1)[1].split("</script>", 1)[0])

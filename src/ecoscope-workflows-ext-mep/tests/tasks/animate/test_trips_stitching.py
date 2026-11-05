"""trajectory_to_trips: segments the segment filter removed are not joined up, but marked as breaks."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from ecoscope_workflows_ext_mep.tasks.animate import _draw_map as dm
from ecoscope_workflows_ext_mep.tasks.animate._draw_map import trajectory_to_trips

T0 = pd.Timestamp("2026-09-09 07:48", tz="UTC")


def _segments(subject: str, fixes: list[tuple[float, float, float]], keep: list[int]) -> list[dict]:
    """The segments between consecutive (hour, lon, lat) fixes, keeping only those numbered in `keep`."""
    rows = []
    for k in keep:
        (h0, x0, y0), (h1, x1, y1) = fixes[k], fixes[k + 1]
        rows.append(
            {
                "groupby_col": subject,
                "subject_name": subject,
                "segment_start": T0 + pd.Timedelta(hours=h0),
                "segment_end": T0 + pd.Timedelta(hours=h1),
                "geometry": LineString([(x0, y0), (x1, y1)]),
            }
        )
    return rows


def _trips(rows: list[dict], **kwargs) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return trajectory_to_trips(gdf, keep_cols=["subject_name"], **kwargs).set_index("groupby_col")


# Moving steadily, then a 96 h stretch the filter removed, then moving again.
STEADY = [(h, 37.0 + 0.01 * h, -2.6) for h in range(4)] + [(100 + h, 37.2 + 0.01 * h, -2.6) for h in range(4)]


def test_unfiltered_track_has_no_breaks():
    trip = _trips(_segments("A", STEADY, keep=list(range(len(STEADY) - 1)))).loc["A"]
    assert len(trip.geometry.coords) == len(STEADY) and trip.breaks == []


def test_removed_stretch_becomes_a_break_not_a_line():
    trip = _trips(_segments("A", STEADY, keep=[0, 1, 2, 4, 5, 6])).loc["A"]
    assert len(trip.geometry.coords) == 8
    assert trip.breaks == [3]  # fix 3 -> fix 4 crosses the removed stretch
    assert trip.timestamps[4] - trip.timestamps[3] == 97 * 3600


def test_short_piece_cut_off_by_a_break_is_dropped():
    # Two test fixes 3 min apart, 193 km away, 5 days before the collar starts moving with the animal.
    fixes = [(0, 38.75279, -3.574993), (0.05, 38.75282, -3.574995)] + [
        (124 + h, 37.26685 + 0.001 * h, -2.679365) for h in range(4)
    ]
    trip = _trips(_segments("IRI", fixes, keep=[0, 2, 3, 4])).loc["IRI"]
    assert [round(x, 2) for x, _ in trip.geometry.coords] == [37.27] * 4
    assert trip.breaks == []


def test_min_piece_fixes_can_keep_short_pieces():
    fixes = [(0, 38.75, -3.57), (0.05, 38.75, -3.58)] + [(124 + h, 37.27 + 0.001 * h, -2.68) for h in range(4)]
    trip = _trips(_segments("IRI", fixes, keep=[0, 2, 3, 4]), min_piece_fixes=1).loc["IRI"]
    assert len(trip.geometry.coords) == 6 and trip.breaks == [1]


def test_short_removed_stretch_is_joined_up():
    # The filter dropped one resting step (fix 3 -> 4, 1 h): no break, the track carries on.
    fixes = [(h, 37.0 + 0.01 * h, -2.6) for h in range(8)]
    trip = _trips(_segments("A", fixes, keep=[0, 1, 2, 4, 5, 6])).loc["A"]
    assert len(trip.geometry.coords) == 8 and trip.breaks == []


def test_break_hours_threshold():
    trip = _trips(_segments("A", STEADY, keep=[0, 1, 2, 4, 5, 6]), min_break_hours=100).loc["A"]
    assert trip.breaks == []  # the 97 h stretch is under the threshold


def test_subject_with_only_short_pieces_keeps_its_latest():
    # Two 2-fix pieces 59 h apart (e.g. test fixes, then a collar that stopped): the latest piece stays.
    fixes = [(0, 37.0, -2.6), (1, 37.01, -2.6), (60, 37.5, -2.6), (61, 37.51, -2.6)]
    trip = _trips(_segments("B", fixes, keep=[0, 2])).loc["B"]
    assert [x for x, _ in trip.geometry.coords] == [37.5, 37.51] and trip.breaks == []


class TestTrackAltitudeModes:
    """drape_trips_on_terrain places tracks as Google Maps 3D altitude modes do."""

    @pytest.fixture
    def trips(self):
        return gpd.GeoDataFrame(
            {"timestamps": [[0, 1, 2]], "breaks": [[]]},
            geometry=[LineString([(37.0, -2.6, 1500), (37.01, -2.6, float("nan")), (37.02, -2.6, 1520)])],
            crs="EPSG:4326",
        )

    @pytest.fixture(autouse=True)
    def flat_ground(self, monkeypatch):
        monkeypatch.setattr(dm, "sample_elevations", lambda lonlats, **kw: [1200.0] * len(lonlats))

    def _z(self, gdf):
        return [round(c[2], 1) for c in gdf.geometry.iloc[0].coords]

    def _drape(self, trips, altitude, densify_m=0.0):
        return dm.drape_trips_on_terrain(trips, dm.TerrainSampling(altitude=altitude, densify_m=densify_m))

    def test_relative_to_ground_is_the_default(self, trips):
        assert isinstance(dm.TerrainSampling().altitude, dm.RelativeToGround)
        assert self._z(self._drape(trips, dm.RelativeToGround(height_m=30))) == [1230.0] * 3

    def test_clamp_to_ground_ignores_recorded_altitude(self, trips):
        assert self._z(self._drape(trips, dm.ClampToGround())) == [1200.0] * 3

    def test_absolute_uses_recorded_altitude_else_a_height_above_the_ground(self, trips):
        out = self._drape(trips, dm.AbsoluteAltitude(fallback_height_m=40))
        assert self._z(out) == [1500.0, 1240.0, 1520.0]  # not 40 m above sea level

    def test_absolute_without_any_altitude_stays_above_the_ground(self, trips):
        trips = trips.set_geometry([LineString([(37.0, -2.6), (37.01, -2.6)])])
        trips["timestamps"] = [[0, 1]]
        assert self._z(self._drape(trips, dm.AbsoluteAltitude())) == [1230.0, 1230.0]

    def test_relative_to_mesh_tags_the_rows_and_keeps_true_heights(self, trips):
        out = self._drape(trips, dm.RelativeToMesh(height_m=15))
        assert self._z(out) == [1215.0] * 3  # what the camera frames
        assert out[dm.MESH_HEIGHT_COL].tolist() == [15.0]

    def test_other_modes_drop_a_stale_mesh_tag(self, trips):
        trips[dm.MESH_HEIGHT_COL] = 15.0
        assert dm.MESH_HEIGHT_COL not in self._drape(trips, dm.ClampToGround()).columns

    def test_densify_follows_the_terrain_between_fixes(self, trips):
        out = self._drape(trips, dm.ClampToGround(), densify_m=100)
        coords = list(out.geometry.iloc[0].coords)
        ts = out["timestamps"].iloc[0]
        assert len(coords) == len(ts) > 20  # ~1.1 km steps -> 12 points each
        assert ts[0] == 0 and ts[-1] == 2 and ts == sorted(ts)
        assert coords[0][:2] == (37.0, -2.6) and coords[-1][:2] == (37.02, -2.6)

    def test_densify_keeps_the_fixes_and_their_times(self, trips):
        out = self._drape(trips, dm.ClampToGround(), densify_m=500)
        coords, ts = list(out.geometry.iloc[0].coords), out["timestamps"].iloc[0]
        fixes = [round(x, 6) for (x, _, _), t in zip(coords, ts) if t in (0, 1, 2)]
        assert fixes == [37.0, 37.01, 37.02]

    def test_densify_leaves_data_gaps_alone_and_renumbers_them(self, trips):
        trips["breaks"] = [[1]]  # no line from fix 1 to fix 2
        out = self._drape(trips, dm.ClampToGround(), densify_m=100)
        coords, ts, (brk,) = list(out.geometry.iloc[0].coords), out["timestamps"].iloc[0], out["breaks"].iloc[0]
        assert (ts[brk], ts[brk + 1]) == (1, 2)  # still one step across the gap
        assert coords[brk + 1][0] == 37.02 and len(coords) == brk + 2

    def test_rows_without_one_timestamp_per_vertex_are_not_densified(self, trips):
        trips["timestamps"] = [[0, 2]]
        out = self._drape(trips, dm.ClampToGround(), densify_m=100)
        assert len(out.geometry.iloc[0].coords) == 3 and out["timestamps"].iloc[0] == [0, 2]

    def test_densify_widens_its_spacing_on_huge_data(self, trips, monkeypatch):
        monkeypatch.setattr(dm, "MAX_DENSIFIED_VERTICES", 10)
        out = self._drape(trips, dm.ClampToGround(), densify_m=1)
        assert len(out.geometry.iloc[0].coords) <= 10

    @pytest.mark.parametrize(
        "legacy, expected",
        [
            ({"altitude_mode": "clamp_to_ground", "offset": 30}, dm.ClampToGround()),
            ({"altitude_mode": "relative_to_ground", "offset": 12}, dm.RelativeToGround(height_m=12)),
            ({"altitude_mode": "absolute", "offset": 50}, dm.AbsoluteAltitude(fallback_height_m=50)),
        ],
    )
    def test_settings_from_before_the_modes_had_their_own_fields(self, legacy, expected):
        assert dm.TerrainSampling(**legacy).altitude == expected

    def test_form_shows_only_each_modes_own_fields(self):
        import inspect

        from pydantic import TypeAdapter

        param = inspect.signature(dm.create_terrain_sampling).parameters
        assert "offset" not in param and "ground_elevation" not in param
        defs = TypeAdapter(param["altitude"].annotation).json_schema()["$defs"]
        fields = {d["title"]: set(d["properties"]) - {"mode"} for d in defs.values()}
        assert fields == {
            "On the ground": set(),
            "Above the ground": {"height_m"},
            "Recorded altitude (where available)": {"fallback_height_m"},
            "Above the 3D terrain": {"height_m"},
        }

    def test_mesh_height_reaches_the_animation_spec(self, trips, monkeypatch):
        out = self._drape(trips, dm.RelativeToMesh(height_m=15))
        out["rgba_color"] = [[255, 0, 0, 255]]
        layer = dm.PydeckLayerDefinition(
            layer_type="TripsLayer",
            layer_style=dm.TripsLayerStyle(get_color="rgba_color"),
            legend=None,
            geodataframe=out,
        )
        html = dm.draw_animated_map(
            geo_layers=dm.animate_layer(layer=layer, animation=dm.TripsAnimation()),
            view_state=dm.ViewState(longitude=37.01, latitude=-2.6, zoom=12),
            timeline=dm.TimelineAnimation(),
        )
        assert '"meshHeight": 15.0' in html

    def test_stitching_keeps_a_recorded_altitude(self):
        rows = [
            {
                "groupby_col": "A",
                "subject_name": "A",
                "segment_start": T0,
                "segment_end": T0 + pd.Timedelta(hours=1),
                "geometry": LineString([(37.0, -2.6, 1500), (37.01, -2.6, 1510)]),
            },
            {
                "groupby_col": "A",
                "subject_name": "A",
                "segment_start": T0 + pd.Timedelta(hours=1),
                "segment_end": T0 + pd.Timedelta(hours=2),
                "geometry": LineString([(37.01, -2.6, 1510), (37.02, -2.6, 1520)]),
            },
        ]
        trip = _trips(rows, min_piece_fixes=1).loc["A"]
        assert [c[2] for c in trip.geometry.coords] == [1500, 1510, 1520]


class TestTerrainGridForTheCamera:
    """draw_animated_map's elevation grid for the video camera's line-of-sight check."""

    @pytest.fixture
    def layers(self):
        gdf = gpd.GeoDataFrame(
            {"timestamps": [[0, 1]]}, geometry=[LineString([(35.0, -1.5), (35.2, -1.3)])], crs="EPSG:4326"
        )
        return [
            dm.PydeckLayerDefinition(
                layer_type="TripsLayer", layer_style=dm.TripsLayerStyle(), legend=None, geodataframe=gdf
            )
        ]

    def test_covers_the_tracks_with_a_margin(self, layers, monkeypatch, real_terrain_grid):
        seen = {}

        def fake(lonlats, **kw):
            seen.update(kw, n=len(lonlats))
            return [1600.0] * len(lonlats)

        monkeypatch.setattr(dm, "sample_elevations", fake)
        g = real_terrain_grid([dm.create_terrain_layer()], layers)
        assert (g["nx"], g["ny"], len(g["z"]), seen["n"]) == (96, 96, 96 * 96, 96 * 96)
        assert g["w"] < 35.0 and g["e"] > 35.2 and g["s"] < -1.5 and g["n"] > -1.3  # ~4 km margin
        assert set(g["z"]) == {1600} and 5 <= seen["zoom"] <= 13
        assert seen["decoder"] == dm.create_terrain_layer().elevation_decoder  # same heights as drawn

    def test_none_without_terrain_or_when_sampling_fails(self, layers, monkeypatch, real_terrain_grid):
        assert real_terrain_grid([], layers) is None

        def offline(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(dm, "sample_elevations", offline)
        assert real_terrain_grid([dm.create_terrain_layer()], layers) is None

    def test_page_carries_it(self, layers, monkeypatch):
        monkeypatch.setattr(
            dm, "_terrain_grid", lambda *a: {"w": 0, "s": 0, "e": 1, "n": 1, "nx": 2, "ny": 2, "z": [1, 2, 3, 4]}
        )
        layers[0].geodataframe["rgba_color"] = [[255, 0, 0, 255]]
        html = dm.draw_animated_map(
            geo_layers=dm.animate_layer(layer=layers[0], animation=dm.TripsAnimation()),
            tile_layers=[dm.create_terrain_layer()],
            view_state=dm.ViewState(longitude=35.1, latitude=-1.4, zoom=10),
            timeline=dm.TimelineAnimation(),
        )
        assert 'window.__terrainGrid = {"w":0,"s":0,"e":1,"n":1,"nx":2,"ny":2,"z":[1,2,3,4]};' in html

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
            {"timestamps": [[0, 1, 2]]},
            geometry=[LineString([(37.0, -2.6, 1500), (37.01, -2.6, float("nan")), (37.02, -2.6, 1520)])],
            crs="EPSG:4326",
        )

    @pytest.fixture(autouse=True)
    def flat_ground(self, monkeypatch):
        monkeypatch.setattr(dm, "sample_elevations", lambda lonlats, **kw: [1200.0] * len(lonlats))

    def _z(self, gdf):
        return [round(c[2], 1) for c in gdf.geometry.iloc[0].coords]

    def test_relative_to_ground_is_the_default(self, trips):
        out = dm.drape_trips_on_terrain(trips, dm.TerrainSampling(offset=30))
        assert self._z(out) == [1230.0] * 3

    def test_clamp_to_ground_ignores_the_offset(self, trips):
        out = dm.drape_trips_on_terrain(trips, dm.TerrainSampling(altitude_mode="clamp_to_ground", offset=30))
        assert self._z(out) == [1200.0] * 3

    def test_absolute_uses_recorded_altitude_else_the_offset(self, trips):
        out = dm.drape_trips_on_terrain(trips, dm.TerrainSampling(altitude_mode="absolute", offset=1000))
        assert self._z(out) == [1500.0, 1000.0, 1520.0]

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

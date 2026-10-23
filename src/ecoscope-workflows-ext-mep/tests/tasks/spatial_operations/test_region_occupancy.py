"""Tests for ecoscope_workflows_ext_mep.tasks.spatial_operations._region_occupancy."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from ecoscope_workflows_ext_mep.tasks.spatial_operations._region_occupancy import compute_region_occupancy

# Squares are laid out in meters around the central meridian of UTM zone 37S (near Nairobi),
# the zone the task picks for them, so their areas come out exact.
CRS = "EPSG:32737"


def _sq(x, y, size=1000):
    x0, y0 = 500_000 + x, 9_850_000 + y
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


def _etd(*ranges, percentile=99.9, subjects=None):
    data = {"percentile": [percentile] * len(ranges)}
    if subjects is not None:
        data["subject_name"] = subjects
    return gpd.GeoDataFrame(data, geometry=list(ranges), crs=CRS)


def _regions(names, geoms, column="name"):
    return gpd.GeoDataFrame({} if column is None else {column: names}, geometry=geoms, crs=CRS)


def _run(etd, regions, **kwargs):
    return compute_region_occupancy(etd_gdf=etd, regions_gdf=regions, **kwargs).set_index("region")


class TestComputeRegionOccupancy:
    def test_occupancy_is_the_share_of_the_region_inside_the_home_range(self):
        # A 1 km² home range covering a quarter of a 4 km² park, and all of a 0.25 km² plot.
        regions = _regions(["Park", "Plot"], [_sq(0, 0, 2000), _sq(0, 0, 500)])

        result = _run(_etd(_sq(0, 0)), regions)

        assert result.loc["Park", "region_area_sqkm"] == pytest.approx(4.0)
        assert result.loc["Park", "area_sqkm"] == pytest.approx(1.0)
        assert result.loc["Park", "occupancy_pct"] == pytest.approx(25.0)
        assert result.loc["Plot", "occupancy_pct"] == pytest.approx(100.0)

    def test_region_outside_the_home_range_is_zero(self):
        result = _run(_etd(_sq(0, 0)), _regions(["Far"], [_sq(5000, 5000)]))

        assert result.loc["Far", "area_sqkm"] == 0.0 and result.loc["Far", "occupancy_pct"] == 0.0

    def test_rows_with_the_same_region_name_are_merged(self):
        regions = _regions(["Park", "Park"], [_sq(0, 0, 500), _sq(1500, 0, 500)])

        result = _run(_etd(_sq(0, 0)), regions)

        assert result.index.tolist() == ["Park"]
        assert result.loc["Park", "region_area_sqkm"] == pytest.approx(0.5)
        assert result.loc["Park", "occupancy_pct"] == pytest.approx(50.0)

    def test_regions_without_area_are_skipped(self):
        line = LineString([(500_000, 9_850_000), (500_500, 9_850_500)])

        result = _run(_etd(_sq(0, 0)), _regions(["Road", "Park"], [line, _sq(0, 0)]))

        assert result.index.tolist() == ["Park"]

    def test_one_block_of_rows_per_subject(self):
        etd = _etd(_sq(0, 0), _sq(2000, 0), subjects=["Luna", "Soutine"])

        result = _run(etd, _regions(["Park"], [_sq(0, 0, 2000)])).loc["Park"]

        assert dict(zip(result["subject_name"], result["occupancy_pct"])) == {"Luna": 25.0, "Soutine": 0.0}

    def test_other_percentiles_are_ignored(self):
        etd = pd.concat([_etd(_sq(0, 0)), _etd(_sq(0, 0, 500), percentile=50.0)])

        result = _run(etd, _regions(["Park"], [_sq(0, 0)]), percentile=50.0)

        assert result.loc["Park", "occupancy_pct"] == pytest.approx(25.0)

    def test_areas_stay_true_away_from_the_equator(self):
        # At 60°N a 0.1° square is ~61.9 km²; Web Mercator would report about 4x that.
        square = Polygon([(10.0, 60.0), (10.1, 60.0), (10.1, 60.1), (10.0, 60.1)])
        etd = gpd.GeoDataFrame({"percentile": [99.9]}, geometry=[square], crs="EPSG:4326")
        regions = gpd.GeoDataFrame({"name": ["All"]}, geometry=[square], crs="EPSG:4326")

        result = _run(etd, regions)

        assert result.loc["All", "region_area_sqkm"] == pytest.approx(61.9, rel=0.01)
        assert result.loc["All", "occupancy_pct"] == pytest.approx(100.0)

    def test_region_names_fall_back_to_a_name_like_column_then_numbers(self):
        titled = _regions(["Park"], [_sq(0, 0)], column="title")
        unnamed = _regions(None, [_sq(0, 0), _sq(1000, 0)], column=None)

        assert _run(_etd(_sq(0, 0)), titled).index.tolist() == ["Park"]
        assert _run(_etd(_sq(0, 0)), unnamed).index.tolist() == ["Region 1", "Region 2"]

    def test_missing_percentile_lists_the_available_ones(self):
        with pytest.raises(ValueError, match=r"no 99.9 percentile.*\[95.0\]"):
            _run(_etd(_sq(0, 0), percentile=95.0), _regions(["A"], [_sq(0, 0)]))

    def test_empty_home_range_raises(self):
        with pytest.raises(ValueError, match="no area"):
            _run(_etd(Polygon()), _regions(["A"], [_sq(0, 0)]))

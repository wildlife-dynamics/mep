"""Tests for ecoscope_workflows_ext_mep.tasks.spatial_operations._assert.

`assert_polygon_types` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, box

from ecoscope_workflows_ext_mep.tasks.spatial_operations._assert import (
    assert_polygon_types,
)


class TestAssertPolygonTypes:
    def test_returns_the_same_frame_when_all_geometries_are_polygons(self):
        gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)]}, crs="EPSG:3857")

        result = assert_polygon_types(gdf)

        assert result is gdf

    def test_multipolygon_is_accepted(self):
        gdf = gpd.GeoDataFrame(
            {"geometry": [MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])]},
            crs="EPSG:3857",
        )

        assert assert_polygon_types(gdf) is gdf

    def test_mixed_polygon_and_multipolygon_is_accepted(self):
        gdf = gpd.GeoDataFrame(
            {"geometry": [box(0, 0, 1, 1), MultiPolygon([box(2, 2, 3, 3)])]},
            crs="EPSG:3857",
        )

        assert assert_polygon_types(gdf) is gdf

    def test_point_geometry_raises_value_error(self):
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:3857")

        with pytest.raises(ValueError, match="Point"):
            assert_polygon_types(gdf)

    def test_linestring_geometry_raises_value_error(self):
        gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 1)])]}, crs="EPSG:3857")

        with pytest.raises(ValueError, match="LineString"):
            assert_polygon_types(gdf)

    def test_a_single_non_polygon_row_fails_even_alongside_valid_polygons(self):
        gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1), Point(5, 5)]}, crs="EPSG:3857")

        with pytest.raises(ValueError):
            assert_polygon_types(gdf)

"""Tests for ecoscope_workflows_ext_mep.tasks.spatial_operations._overlay.

`overlay_gdf` is registered via `wt_registry.register()`, a no-op
decorator at call time, so it is called directly as plain Python.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from ecoscope_workflows_ext_mep.tasks.spatial_operations._overlay import (
    overlay_gdf,
)


class TestOverlayGdf:
    def test_intersection_returns_the_overlapping_region(self):
        left = gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 2)]}, crs="EPSG:3857")
        right = gpd.GeoDataFrame({"geometry": [box(1, 1, 3, 3)]}, crs="EPSG:3857")

        result = overlay_gdf(left, right, how="intersection")

        assert len(result) == 1
        assert result.geometry.iloc[0].bounds == (1.0, 1.0, 2.0, 2.0)

    def test_difference_returns_the_non_overlapping_part_of_left(self):
        left = gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 2)]}, crs="EPSG:3857")
        right = gpd.GeoDataFrame({"geometry": [box(1, 0, 2, 2)]}, crs="EPSG:3857")

        result = overlay_gdf(left, right, how="difference")

        assert len(result) == 1
        assert result.geometry.iloc[0].area == pytest.approx(2.0)  # the remaining 1x2 strip

    def test_no_overlap_returns_empty_intersection(self):
        left = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:3857")
        right = gpd.GeoDataFrame({"geometry": [box(10, 10, 11, 11)]}, crs="EPSG:3857")

        result = overlay_gdf(left, right, how="intersection")

        assert len(result) == 0

    def test_default_how_is_intersection(self):
        left = gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 2)]}, crs="EPSG:3857")
        right = gpd.GeoDataFrame({"geometry": [box(1, 1, 3, 3)]}, crs="EPSG:3857")

        result = overlay_gdf(left, right)

        assert result.geometry.iloc[0].bounds == (1.0, 1.0, 2.0, 2.0)

    def test_mismatched_crs_raises_value_error_instead_of_silently_overlaying_raw_coords(self):
        left = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:3857")
        right = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")

        with pytest.raises(ValueError, match="CRS mismatch"):
            overlay_gdf(left, right)

    def test_both_frames_with_no_crs_is_allowed(self):
        left = gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 2)]})
        right = gpd.GeoDataFrame({"geometry": [box(1, 1, 3, 3)]})

        result = overlay_gdf(left, right, how="intersection")

        assert len(result) == 1

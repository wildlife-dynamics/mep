"""Tests for ecoscope_workflows_ext_mep.tasks.spatial_operations._opacity.

`set_spatial_features_opacity` is registered via `wt_registry.register()`,
a no-op decorator at call time, so it is called directly as plain Python.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from ecoscope_workflows_ext_mep.tasks.spatial_operations._opacity import (
    set_spatial_features_opacity,
)


class TestSetSpatialFeaturesOpacity:
    def test_overwrites_the_alpha_channel_of_fill_and_line_colors(self):
        gdf = gpd.GeoDataFrame(
            {
                "get_fill_color": [[10, 20, 30, 999]],
                "get_line_color": [[1, 2, 3, 999]],
                "geometry": [box(0, 0, 1, 1)],
            },
            crs="EPSG:3857",
        )

        result = set_spatial_features_opacity(gdf, fill_opacity=0.5, line_opacity=1.0)

        assert result["get_fill_color"].iloc[0] == [10, 20, 30, 127]
        assert result["get_line_color"].iloc[0] == [1, 2, 3, 255]

    def test_fill_opacity_zero_makes_fill_fully_transparent(self):
        gdf = gpd.GeoDataFrame(
            {"get_fill_color": [[10, 20, 30, 255]], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:3857",
        )

        result = set_spatial_features_opacity(gdf, fill_opacity=0.0)

        assert result["get_fill_color"].iloc[0][3] == 0

    def test_missing_color_columns_are_left_untouched(self):
        gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:3857")

        result = set_spatial_features_opacity(gdf, fill_opacity=0.5, line_opacity=0.5)

        assert "get_fill_color" not in result.columns
        assert "get_line_color" not in result.columns

    def test_non_list_color_values_are_left_untouched(self):
        gdf = gpd.GeoDataFrame(
            {"get_fill_color": [None], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:3857",
        )

        result = set_spatial_features_opacity(gdf, fill_opacity=0.5)

        assert result["get_fill_color"].iloc[0] is None

    def test_does_not_mutate_the_input_dataframe(self):
        gdf = gpd.GeoDataFrame(
            {"get_fill_color": [[10, 20, 30, 255]], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:3857",
        )

        set_spatial_features_opacity(gdf, fill_opacity=0.0)

        assert gdf["get_fill_color"].iloc[0] == [10, 20, 30, 255]

    def test_defaults_are_fill_transparent_and_line_opaque(self):
        gdf = gpd.GeoDataFrame(
            {
                "get_fill_color": [[10, 20, 30, 255]],
                "get_line_color": [[1, 2, 3, 255]],
                "geometry": [box(0, 0, 1, 1)],
            },
            crs="EPSG:3857",
        )

        result = set_spatial_features_opacity(gdf)

        assert result["get_fill_color"].iloc[0][3] == 0
        assert result["get_line_color"].iloc[0][3] == 255

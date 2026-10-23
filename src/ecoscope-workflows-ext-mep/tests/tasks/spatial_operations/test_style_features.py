"""Tests for ecoscope_workflows_ext_mep.tasks.spatial_operations._style_features."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from ecoscope_workflows_ext_mep.tasks.spatial_operations._style_features import style_spatial_features


class _Client:
    server = "https://site.pamdas.org/"


def _features():
    """Two features shaped like get_features output: a park with native EarthRanger colours and an icon point."""
    return gpd.GeoDataFrame(
        {
            "name": ["Mara Park", "Main Gate"],
            "feature_type": ["Park", "Outposts"],
            "feature_set": ["Boundaries", "POIs"],
            "fill": ["#555555", None],
            "fill-opacity": [0.2, None],
            "stroke": ["#000000", None],
            "stroke-width": [2, None],
            "image": [None, "/static/gate.svg"],
            "width": [None, 20],
        },
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)]), Point(0.5, 0.5)],
        crs="EPSG:4326",
    )


class TestStyleSpatialFeatures:
    def test_native_earthranger_colours_and_icons_are_used(self):
        styled = style_spatial_features(client=_Client(), features=_features())

        assert styled.loc[0, "get_fill_color"] == [85, 85, 85, 51]
        assert styled.loc[1, "icon_url"] == "https://site.pamdas.org/static/gate.svg"
        assert styled.loc[1, "icon_size"] == 20

    def test_legend_groups_by_feature_type_by_default(self):
        styled = style_spatial_features(client=_Client(), features=_features(), legend_title="Areas")

        assert styled["legend_label"].tolist() == ["Park", "Outposts"]
        assert set(styled["legend_title"]) == {"Areas"}

    def test_legend_can_group_by_name_or_set(self):
        by_name = style_spatial_features(client=_Client(), features=_features(), group_by="name")
        by_set = style_spatial_features(client=_Client(), features=_features(), group_by="feature_set")

        assert by_name["legend_label"].tolist() == ["Mara Park", "Main Gate"]
        assert by_set["legend_label"].tolist() == ["Boundaries", "POIs"]

    def test_style_overrides_native_colours(self):
        style = [{"polygon": [{"fill_color": ["#FFA500"], "fill_opacity": 0.5}]}]

        styled = style_spatial_features(client=_Client(), features=_features(), style=style)

        assert styled.loc[0, "get_fill_color"] == [255, 165, 0, 127]

    def test_no_features_returns_an_empty_frame(self):
        assert style_spatial_features(client=_Client(), features=pd.DataFrame()).empty

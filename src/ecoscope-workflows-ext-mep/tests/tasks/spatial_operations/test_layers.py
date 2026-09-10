"""Tests for ecoscope_workflows_ext_mep.tasks.spatial_operations._layers.

`get_spatial_features` and the query classes' `.get()` methods are
exercised against a small fake EarthRanger client -- a stub object that
records which URLs it was asked for and returns canned JSON, rather than
a deep mock -- following the pattern used in tests/tasks/io/test_earthranger.py.
`get_spatial_features` itself is registered via `wt_registry.register()`,
a no-op decorator at call time, so it is called directly as plain Python.
"""

from __future__ import annotations

import pytest

from ecoscope_workflows_ext_mep.tasks.spatial_operations._layers import (
    EarthRangerSource,
    FeatureIdQuery,
    FeatureSetQuery,
    FeatureTypeQuery,
    get_spatial_features,
)


class _FakeClient:
    """Records the URLs it was asked for and returns canned JSON per URL."""

    server = "https://example.earthranger.com"

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []

    def _get(self, url: str):
        self.calls.append(url)
        return self.routes[url]


def _polygon_feature(**properties) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "properties": properties,
    }


class TestFeatureIdQuery:
    def test_loads_a_single_feature_by_id_and_resolves_its_type_name(self):
        client = _FakeClient(
            {
                "feature/abc-123/": {"features": [_polygon_feature(title="Zone A", feature_type="ft-1")]},
                "featureset/": {"features": [{"id": "fs-1", "types": [{"id": "ft-1", "name": "Conservancy"}]}]},
            }
        )

        gdf = FeatureIdQuery(feature_id="abc-123").get(client)

        assert gdf["title"].tolist() == ["Zone A"]
        assert gdf["type_name"].tolist() == ["Conservancy"]

    def test_missing_feature_returns_empty_frame(self):
        client = _FakeClient({"feature/missing/": {"features": []}})

        gdf = FeatureIdQuery(feature_id="missing").get(client)

        assert len(gdf) == 0

    def test_feature_without_a_feature_type_property_raises_key_error(self):
        # Documents current behavior: `feature_type` is accessed
        # unconditionally, so a feature response lacking that property
        # (e.g. it was never set in EarthRanger) raises rather than
        # degrading to a missing/null type_name.
        client = _FakeClient(
            {
                "feature/abc-123/": {"features": [_polygon_feature(title="Zone A")]},
                "featureset/": {"features": []},
            }
        )

        with pytest.raises(KeyError, match="feature_type"):
            FeatureIdQuery(feature_id="abc-123").get(client)


class TestFeatureSetQuery:
    def test_loads_all_features_in_the_named_featureset(self):
        client = _FakeClient(
            {
                "featureset/": {"features": [{"id": "fs-1", "name": "Boundaries"}]},
                "featureset/fs-1/": {"features": [_polygon_feature(title="Zone A")]},
            }
        )

        gdf = FeatureSetQuery(featureset_name="Boundaries").get(client)

        assert gdf["title"].tolist() == ["Zone A"]

    def test_unknown_featureset_name_raises_value_error_listing_available_names(self):
        client = _FakeClient({"featureset/": {"features": [{"id": "fs-1", "name": "Boundaries"}]}})

        with pytest.raises(ValueError, match="Boundaries"):
            FeatureSetQuery(featureset_name="Does Not Exist").get(client)

    def test_empty_result_returns_an_empty_geodataframe_not_none(self):
        client = _FakeClient(
            {
                "featureset/": {"features": [{"id": "fs-1", "name": "Boundaries"}]},
                "featureset/fs-1/": {"features": []},
            }
        )

        gdf = FeatureSetQuery(featureset_name="Boundaries").get(client)

        assert len(gdf) == 0


class TestFeatureTypeQuery:
    def test_loads_only_features_matching_the_requested_type(self):
        client = _FakeClient(
            {
                "featureclass/": [{"name": "Conservancy", "feature_set_id": "fs-2"}],
                "featureset/fs-2/": {
                    "features": [
                        _polygon_feature(title="A", type_name="Conservancy"),
                        _polygon_feature(title="B", type_name="Other"),
                    ]
                },
            }
        )

        gdf = FeatureTypeQuery(feature_type="Conservancy").get(client)

        assert gdf["title"].tolist() == ["A"]

    def test_unknown_feature_type_raises_value_error(self):
        client = _FakeClient({"featureclass/": []})

        with pytest.raises(ValueError, match="not found"):
            FeatureTypeQuery(feature_type="Conservancy").get(client)

    def test_feature_type_not_linked_to_a_featureset_raises_value_error(self):
        client = _FakeClient({"featureclass/": [{"name": "Conservancy", "feature_set_id": None}]})

        with pytest.raises(ValueError, match="not linked"):
            FeatureTypeQuery(feature_type="Conservancy").get(client)


class TestEarthRangerSource:
    def test_no_query_returns_an_empty_geodataframe(self):
        source = EarthRangerSource()

        result = source.get(_FakeClient({}))

        assert len(result) == 0


class TestGetSpatialFeatures:
    def test_no_source_returns_an_empty_dataframe(self):
        result = get_spatial_features(client=_FakeClient({}), source=None)

        assert len(result) == 0

    def test_configured_source_applies_legend_title_and_returns_styled_features(self):
        # A `fill` property is required for `_apply_geo_style` to derive
        # `get_fill_color` -- and thus `legend_title`/`legend_label` --
        # from EarthRanger's own styling in the absence of a `LayerStyle`
        # override.
        client = _FakeClient(
            {
                "feature/abc-123/": {
                    "features": [_polygon_feature(title="Zone A", feature_type="ft-1", fill="#ff0000")]
                },
                "featureset/": {"features": [{"id": "fs-1", "types": [{"id": "ft-1", "name": "Conservancy"}]}]},
            }
        )

        result = get_spatial_features(
            client=client,
            source={"query": {"feature_id": "abc-123"}},
            legend_title="Legend",
        )

        assert result["title"].tolist() == ["Zone A"]
        assert result["legend_title"].tolist() == ["Legend"]

    def test_source_whose_query_returns_no_features_yields_an_empty_dataframe(self):
        client = _FakeClient({"feature/missing/": {"features": []}})

        result = get_spatial_features(client=client, source={"query": {"feature_id": "missing"}})

        assert len(result) == 0

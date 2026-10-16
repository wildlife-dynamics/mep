"""Tests for the feature tasks in ecoscope_workflows_ext_mep.tasks.io._earthranger.

The EarthRanger client is a stub that serves canned `featureset`, `featureset/<id>` and
`spatialfeature/<id>` responses, shaped like the real API's.
"""

from __future__ import annotations

import pytest
from erclient.client import ERClientNotFound
from pydantic import TypeAdapter, ValidationError

from ecoscope_workflows_ext_mep.tasks.io._earthranger import (
    FEATURE_COLUMNS,
    FeatureSelection,
    FeaturesById,
    FeaturesBySet,
    FeaturesByType,
    get_features,
    get_features_by_id,
    get_features_by_set,
    get_features_by_type,
)

BOUNDARIES = {
    "id": "set-b",
    "name": "Boundaries",
    "types": [
        {"id": "t-park", "name": "Park", "feature_count": 1},
        {"id": "t-block", "name": "Block", "feature_count": 1},
    ],
}
POIS = {"id": "set-p", "name": "POIs", "types": [{"id": "t-camp", "name": "Camp", "feature_count": 1}]}
RIVERS = {"id": "set-r", "name": "Rivers", "types": []}


def _feature(fid, type_id, geometry, **props):
    return {
        "type": "Feature",
        "id": fid,
        "properties": {"feature_type": type_id, "feature_geometry_webmercator": "SRID=3857;...", **props},
        "geometry": geometry,
    }


def _collection(*features):
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": list(features),
    }


SQUARE = {"type": "Polygon", "coordinates": [[[35, -1], [35.1, -1], [35.1, -1.1], [35, -1]]]}
POINT = {"type": "Point", "coordinates": [35.05, -1.05]}
PARK = _feature("f-park", "t-park", SQUARE, title="Mara Park", type_name="Park", pk="f-park", description="A park")
BLOCK = _feature("f-block", "t-block", SQUARE, title="Block 1", type_name="Block")
CAMP = _feature("f-camp", "t-camp", POINT, name="Main Camp", attributes={"beds": 4})

RESPONSES = {
    "featureset": {"features": [BOUNDARIES, POIS, RIVERS]},
    "featureset/set-b": _collection(PARK, BLOCK),
    "featureset/set-p": _collection(CAMP),
    "spatialfeature/f-park": _collection(PARK),
    "spatialfeature/f-camp": _collection(CAMP),
}


class _FakeClient:
    def __init__(self):
        self.calls: list[str] = []

    def _get(self, path, **kwargs):
        self.calls.append(path)
        if path not in RESPONSES:
            raise ERClientNotFound(path)
        return RESPONSES[path]


@pytest.fixture
def client():
    return _FakeClient()


class TestGetFeaturesBySet:
    def test_named_sets_become_one_row_per_feature(self, client):
        gdf = get_features_by_set(client=client, feature_set_names=["Boundaries"])

        assert list(gdf["id"]) == ["f-park", "f-block"]
        assert list(gdf["name"]) == ["Mara Park", "Block 1"]
        assert list(gdf["feature_type"]) == ["Park", "Block"]
        assert set(gdf["feature_set"]) == {"Boundaries"} and set(gdf["feature_set_id"]) == {"set-b"}
        assert gdf.crs.to_epsg() == 4326 and list(gdf.geom_type) == ["Polygon", "Polygon"]
        assert list(gdf.columns[: len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS

    def test_redundant_properties_are_dropped(self, client):
        gdf = get_features_by_set(client=client, feature_set_names=["Boundaries"])

        assert not {"feature_geometry_webmercator", "title", "type_name", "pk"} & set(gdf.columns)

    def test_no_names_fetches_every_set_and_skips_sets_without_types(self, client):
        gdf = get_features_by_set(client=client)

        assert list(gdf["id"]) == ["f-park", "f-block", "f-camp"]
        assert "featureset/set-r" not in client.calls

    def test_unknown_set_name_lists_the_available_ones(self, client):
        with pytest.raises(ValueError, match="Unknown feature set name.*Nope.*Available: Boundaries, POIs, Rivers"):
            get_features_by_set(client=client, feature_set_names=["Nope"])

    def test_empty_set_returns_an_empty_gdf_with_the_standard_columns(self, client):
        gdf = get_features_by_set(client=client, feature_set_names=["Rivers"])

        assert gdf.empty and list(gdf.columns[: len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS

    def test_raise_on_empty(self, client):
        with pytest.raises(ValueError, match="No data returned from EarthRanger"):
            get_features_by_set(client=client, feature_set_names=["Rivers"], raise_on_empty=True)


class TestGetFeaturesByType:
    def test_only_the_chosen_types_are_kept(self, client):
        gdf = get_features_by_type(client=client, feature_type_names=["Park", "Camp"])

        assert sorted(gdf["id"]) == ["f-camp", "f-park"]
        assert dict(zip(gdf["id"], gdf["feature_set"])) == {"f-park": "Boundaries", "f-camp": "POIs"}

    def test_only_sets_holding_the_types_are_fetched(self, client):
        get_features_by_type(client=client, feature_type_names=["Camp"])

        assert client.calls == ["featureset", "featureset/set-p"]

    def test_unknown_type_name_raises(self, client):
        with pytest.raises(ValueError, match="Unknown feature type name.*Nope"):
            get_features_by_type(client=client, feature_type_names=["Nope"])


class TestGetFeaturesById:
    def test_features_are_fetched_by_id_with_type_and_set(self, client):
        gdf = get_features_by_id(client=client, feature_ids=["f-park", " f-camp "])

        assert list(gdf["id"]) == ["f-park", "f-camp"]
        assert list(gdf["feature_type"]) == ["Park", "Camp"]
        assert list(gdf["feature_set"]) == ["Boundaries", "POIs"]
        assert gdf.loc[1, "attributes"] == {"beds": 4}

    def test_unknown_id_raises_a_clear_error(self, client):
        with pytest.raises(ValueError, match="No EarthRanger feature with id missing"):
            get_features_by_id(client=client, feature_ids=["missing"])

    def test_no_ids_returns_empty_without_calling_earthranger(self, client):
        gdf = get_features_by_id(client=client, feature_ids=[])

        assert gdf.empty and client.calls == []


class TestGetFeatures:
    @pytest.mark.parametrize(
        "selection,ids",
        [
            ({}, ["f-park", "f-block", "f-camp"]),
            ({"feature_set_names": ["POIs"]}, ["f-camp"]),
            ({"feature_type_names": ["Park"]}, ["f-park"]),
            ({"feature_ids": ["f-camp"]}, ["f-camp"]),
        ],
    )
    def test_each_selection_fetches_its_features(self, client, selection, ids):
        selection = TypeAdapter(FeatureSelection).validate_python(selection)

        assert list(get_features(client=client, selection=selection)["id"]) == ids

    def test_default_is_every_feature_set(self, client):
        assert len(get_features(client=client)) == 3

    def test_selection_matches_exactly_one_option(self):
        adapter = TypeAdapter(FeatureSelection)

        assert isinstance(adapter.validate_python({"feature_type_names": ["Park"]}), FeaturesByType)
        assert isinstance(adapter.validate_python({"feature_ids": ["x"]}), FeaturesById)
        assert isinstance(adapter.validate_python({}), FeaturesBySet)
        with pytest.raises(ValidationError):
            adapter.validate_python({"feature_type_names": ["Park"], "feature_ids": ["x"]})

    def test_form_offers_the_three_options(self):
        schema = TypeAdapter(FeatureSelection).json_schema()

        titles = [schema["$defs"][ref["$ref"].split("/")[-1]]["title"] for ref in schema["anyOf"]]
        assert titles == ["Feature sets", "Feature types", "Feature IDs"]

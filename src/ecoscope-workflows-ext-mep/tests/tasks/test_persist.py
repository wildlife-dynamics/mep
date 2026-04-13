import json

import geopandas as gpd
import pytest
from shapely.geometry import Point

from ecoscope_workflows_ext_mep.tasks._persist import gdf_to_geojson


@pytest.fixture
def sample_gdf():
    return gpd.GeoDataFrame(
        {"value": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
        crs="EPSG:4326",
    )


def test_basic_serialization(sample_gdf, tmp_path):
    path = gdf_to_geojson(df=sample_gdf, root_path=str(tmp_path))
    assert path.endswith(".geojson")
    with open(path) as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 3


def test_custom_filename(sample_gdf, tmp_path):
    path = gdf_to_geojson(df=sample_gdf, root_path=str(tmp_path), filename="my_output")
    assert path.endswith("my_output.geojson")


def test_tuple_columns_serialized_as_lists(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "color": [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            "label": ["a", "b", "c"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    path = gdf_to_geojson(df=gdf, root_path=str(tmp_path))
    with open(path) as f:
        data = json.load(f)
    colors = [f["properties"]["color"] for f in data["features"]]
    labels = [f["properties"]["label"] for f in data["features"]]
    assert colors == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    assert labels == ["a", "b", "c"]


def test_tuple_column_with_nan_values(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "color": [(255, 0, 0), None, (0, 0, 255)],
            "label": ["a", "b", "c"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    path = gdf_to_geojson(df=gdf, root_path=str(tmp_path))
    with open(path) as f:
        data = json.load(f)
    colors = [f["properties"]["color"] for f in data["features"]]
    labels = [f["properties"]["label"] for f in data["features"]]
    assert colors == [[255, 0, 0], None, [0, 0, 255]]
    assert labels == ["a", "b", "c"]


def test_deterministic_hash_for_same_df(sample_gdf, tmp_path):
    path1 = gdf_to_geojson(df=sample_gdf, root_path=str(tmp_path))
    path2 = gdf_to_geojson(df=sample_gdf, root_path=str(tmp_path))
    assert path1 == path2


def test_different_hash_for_different_df(tmp_path):
    gdf1 = gpd.GeoDataFrame({"value": [1], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    gdf2 = gpd.GeoDataFrame({"value": [2], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    path1 = gdf_to_geojson(df=gdf1, root_path=str(tmp_path))
    path2 = gdf_to_geojson(df=gdf2, root_path=str(tmp_path))
    assert path1 != path2

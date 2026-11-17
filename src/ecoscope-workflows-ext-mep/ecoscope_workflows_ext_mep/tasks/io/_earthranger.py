from typing import Annotated, cast

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.connections import EarthRangerClient
from pydantic import BaseModel, ConfigDict, Field
from wt_registry import register

# Columns every feature GeoDataFrame starts with, whichever endpoint the features came from.
FEATURE_COLUMNS = [
    "id",
    "name",
    "feature_type",
    "feature_type_id",
    "feature_set",
    "feature_set_id",
    "description",
    "external_id",
    "attributes",
]
# A web-mercator WKT copy of every geometry; large and redundant with `geometry`.
_DROPPED_PROPERTIES = ["feature_geometry_webmercator"]


def _feature_sets(client: EarthRangerClient) -> list[dict]:
    """Every feature set on the site, each listing its feature types as `{id, name, feature_count}`."""
    return client._get("featureset")["features"]


def _feature_types(feature_sets: list[dict]) -> dict[str, tuple[dict, dict]]:
    """Feature type id -> (feature type, the feature set it belongs to)."""
    return {ftype["id"]: (ftype, fset) for fset in feature_sets for ftype in fset.get("types", [])}


def _pick_by_name(items: list[dict], names: list[str], kind: str) -> list[dict]:
    """The items whose name is one of `names`; raises listing the valid names if any are unknown."""
    wanted = {n.strip() for n in names}
    found = [item for item in items if item.get("name") in wanted]
    missing = wanted - {item["name"] for item in found}
    if missing:
        available = ", ".join(sorted({item["name"] for item in items})) or "none"
        raise ValueError(f"Unknown {kind} name(s): {', '.join(sorted(missing))}. Available: {available}")
    return found


def _features_to_gdf(collection: dict, feature_types: dict[str, tuple[dict, dict]]):
    """A GeoJSON FeatureCollection from EarthRanger as a GeoDataFrame in EPSG:4326, one row per feature,
    named after its feature type and feature set."""
    import geopandas as gpd
    from shapely.geometry import shape

    rows, geometries = [], []
    for feature in collection.get("features", []):
        props = {k: v for k, v in (feature.get("properties") or {}).items() if k not in _DROPPED_PROPERTIES}
        type_id = props.pop("feature_type", None)
        ftype, fset = feature_types.get(type_id, ({}, {}))
        title, type_name, pk = props.pop("title", None), props.pop("type_name", None), props.pop("pk", None)
        rows.append(
            {
                "id": feature.get("id") or pk,
                "name": props.pop("name", None) or title or "",
                "feature_type": ftype.get("name") or type_name,
                "feature_type_id": type_id,
                "feature_set": fset.get("name"),
                "feature_set_id": fset.get("id"),
                "description": props.pop("description", None) or "",
                "external_id": props.pop("external_id", None) or "",
                "attributes": props.pop("attributes", None) or {},
                **props,
            }
        )
        geometries.append(shape(feature["geometry"]) if feature.get("geometry") else None)

    crs = ((collection.get("crs") or {}).get("properties") or {}).get("name") or "EPSG:4326"
    gdf = gpd.GeoDataFrame(rows or None, columns=None if rows else FEATURE_COLUMNS, geometry=geometries, crs=crs)
    return gdf.to_crs(4326)


def _concat(gdfs: list):
    import geopandas as gpd
    import pandas as pd

    gdfs = [g for g in gdfs if not g.empty]
    if not gdfs:
        return gpd.GeoDataFrame(columns=FEATURE_COLUMNS, geometry=[], crs=4326)
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=4326)


def _check_empty(gdf, raise_on_empty: bool, task: str):
    if raise_on_empty and gdf.empty:
        raise ValueError(f"No data returned from EarthRanger for {task}")
    return cast(AnyGeoDataFrame, gdf)


@register()
def get_features_by_set(
    client: EarthRangerClient,
    feature_set_names: Annotated[
        list[str] | None,
        Field(description="Names of the feature sets to fetch, e.g. Boundaries. Leave empty to fetch every set."),
    ] = None,
    raise_on_empty: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="Whether to abort the workflow if the chosen feature sets contain no features.",
        ),
    ] = False,
) -> AnyGeoDataFrame:
    """Fetch every feature in the chosen EarthRanger feature sets as a GeoDataFrame,
    with each feature's type and set."""

    sets = _feature_sets(client)
    if feature_set_names:
        sets = _pick_by_name(sets, feature_set_names, "feature set")
    types = _feature_types(sets)
    gdf = _concat([_features_to_gdf(client._get(f"featureset/{s['id']}"), types) for s in sets if s.get("types")])
    return _check_empty(gdf, raise_on_empty, "get_features_by_set")


@register()
def get_features_by_type(
    client: EarthRangerClient,
    feature_type_names: Annotated[
        list[str],
        Field(description="Names of the feature types to fetch, e.g. Park or Census Block."),
    ],
    raise_on_empty: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="Whether to abort the workflow if the chosen feature types have no features.",
        ),
    ] = False,
) -> AnyGeoDataFrame:
    """Fetch every feature of the chosen EarthRanger feature types as a GeoDataFrame,
    with each feature's type and set."""

    sets = _feature_sets(client)
    types = _feature_types(sets)
    chosen = _pick_by_name([ftype for ftype, _ in types.values()], feature_type_names, "feature type")
    chosen_ids = {ftype["id"] for ftype in chosen}
    set_ids = {types[type_id][1]["id"] for type_id in chosen_ids}

    gdfs = []
    for set_id in set_ids:
        gdf = _features_to_gdf(client._get(f"featureset/{set_id}"), types)
        gdfs.append(gdf[gdf["feature_type_id"].isin(chosen_ids)])
    return _check_empty(_concat(gdfs), raise_on_empty, "get_features_by_type")


@register()
def get_features_by_id(
    client: EarthRangerClient,
    feature_ids: Annotated[list[str], Field(description="IDs of the EarthRanger features to fetch.")],
    raise_on_empty: Annotated[
        bool,
        AdvancedField(default=False, description="Whether to abort the workflow if no features are returned."),
    ] = False,
) -> AnyGeoDataFrame:
    """Fetch the EarthRanger features with the given IDs as a GeoDataFrame, with each feature's type and set."""
    from erclient.client import ERClientNotFound

    types = _feature_types(_feature_sets(client)) if feature_ids else {}
    gdfs = []
    for feature_id in feature_ids:
        try:
            collection = client._get(f"spatialfeature/{feature_id.strip()}")
        except ERClientNotFound as e:
            raise ValueError(f"No EarthRanger feature with id {feature_id}") from e
        gdfs.append(_features_to_gdf(collection, types))
    return _check_empty(_concat(gdfs), raise_on_empty, "get_features_by_id")


class FeaturesBySet(BaseModel):
    model_config = ConfigDict(title="Feature sets")
    # An empty list (not None) means every set, so the form's list starts empty rather than null.
    feature_set_names: Annotated[
        list[str],
        Field(
            default=[],
            title="Feature set names",
            description="Names of the feature sets to fetch, e.g. Boundaries. Leave empty to fetch every set.",
        ),
    ] = []


class FeaturesByType(BaseModel):
    model_config = ConfigDict(title="Feature types")
    feature_type_names: Annotated[
        list[str],
        Field(
            title="Feature type names", description="Names of the feature types to fetch, e.g. Park or Census Block."
        ),
    ]


class FeaturesById(BaseModel):
    model_config = ConfigDict(title="Feature IDs")
    feature_ids: Annotated[
        list[str], Field(title="Feature IDs", description="IDs of the EarthRanger features to fetch.")
    ]


# The options don't forbid extra fields (no `additionalProperties: false`): Ecoscope Desktop validates with
# AJV's removeAdditional, which would then delete the list from the form data while RJSF tries each option.
# Pydantic still picks the option whose field is given.
FeatureSelection = FeaturesBySet | FeaturesByType | FeaturesById


@register()
def get_features(
    client: EarthRangerClient,
    selection: Annotated[
        FeatureSelection,
        Field(title="Select features by", description="Fetch features by feature set, by feature type, or by ID."),
    ] = FeaturesBySet(),
    raise_on_empty: Annotated[
        bool,
        AdvancedField(default=False, description="Whether to abort the workflow if no features are returned."),
    ] = False,
) -> AnyGeoDataFrame:
    """Fetch EarthRanger features as a GeoDataFrame, chosen by feature set, feature type or feature ID,
    with each feature's type and set."""

    if isinstance(selection, FeaturesByType):
        return get_features_by_type(client, selection.feature_type_names, raise_on_empty)
    if isinstance(selection, FeaturesById):
        return get_features_by_id(client, selection.feature_ids, raise_on_empty)
    return get_features_by_set(client, selection.feature_set_names, raise_on_empty)

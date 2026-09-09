from wt_registry import register
from ecoscope.platform.annotations import AnyGeoDataFrame

@register()
def assert_polygon_types(gdf: AnyGeoDataFrame) -> AnyGeoDataFrame:
    """
    Assert all remaining geometries are
    Polygon or MultiPolygon.
    """
    geom_types = set(gdf.geometry.geom_type.unique())
    invalid = geom_types - {"Polygon", "MultiPolygon"}
    if invalid:
        raise ValueError(
            f"Invalid geometry types: {invalid}. "
            f"Only Polygon and MultiPolygon are supported."
        )
    return gdf
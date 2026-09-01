import geopandas as gpd
from pydantic import Field
from wt_registry import register
from typing import Annotated, Any, cast
from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def set_gdf_crs(
    gdf: Annotated[AnyGeoDataFrame, Field(description="Input GeoDataFrame.")],
    target_crs: Annotated[
        Any,
        Field(
            description="The target CRS. Can be a string (e.g., 'EPSG:4326'), "
            "an integer (e.g., 4326), or any object accepted by `pyproj.CRS.from_user_input`."
        ),
    ],
) -> AnyGeoDataFrame:
    result = cast(gpd.GeoDataFrame, gdf).set_crs(target_crs)
    return cast(AnyGeoDataFrame, result)
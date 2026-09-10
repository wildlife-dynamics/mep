from pydantic import Field
from typing import Annotated, cast
from wt_registry import register
from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def set_spatial_features_opacity(
    gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="Styled GeoDataFrame from get_spatial_features.", exclude=True),
    ],
    fill_opacity: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Fill opacity for polygon interiors. Set to 0 to show outlines only.",
        ),
    ] = 0.0,
    line_opacity: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Opacity of polygon borders from 0 (transparent) to 1 (fully opaque).",
        ),
    ] = 1.0,
) -> AnyGeoDataFrame:
    """Overwrite the alpha channel of fill and line color columns to control polygon styling."""
    gdf = gdf.copy()
    fill_alpha = int(fill_opacity * 255)
    line_alpha = int(line_opacity * 255)
    if "get_fill_color" in gdf.columns:
        gdf["get_fill_color"] = gdf["get_fill_color"].apply(
            lambda c: c[:3] + [fill_alpha] if isinstance(c, list) and len(c) >= 3 else c
        )
    if "get_line_color" in gdf.columns:
        gdf["get_line_color"] = gdf["get_line_color"].apply(
            lambda c: c[:3] + [line_alpha] if isinstance(c, list) and len(c) >= 3 else c
        )
    return cast(AnyGeoDataFrame, gdf)

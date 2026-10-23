from typing import Annotated, cast

import pandas as pd
from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame, EmptyDataFrame
from ecoscope.platform.connections import EarthRangerClient
from ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_features import (  # type: ignore[import-untyped]
    LayerStyle,
    _apply_geo_style,
)
from pydantic import Field
from wt_registry import register


@register()
def style_spatial_features(
    client: EarthRangerClient,
    features: Annotated[
        AnyGeoDataFrame | EmptyDataFrame,
        Field(description="Features from get_features (or get_features_by_set/type/id).", exclude=True),
    ],
    style: Annotated[
        list[LayerStyle],
        Field(
            default=[],
            max_length=1,
            description="Optional: Override how EarthRanger spatial features are rendered on the map.",
        ),
    ] = [],
    group_by: Annotated[
        str,
        AdvancedField(
            default="feature_type",
            description="Column used to group features in the map legend.",
            json_schema_extra={
                "oneOf": [
                    {"const": "feature_type", "title": "Feature Type"},
                    {"const": "name", "title": "Feature Name"},
                    {"const": "feature_set", "title": "Feature Set"},
                ]
            },
        ),
    ] = "feature_type",
    legend_title: Annotated[
        str,
        AdvancedField(default="", description="Label shown in the map legend e.g. 'Park Boundary'."),
    ] = "",
) -> AnyGeoDataFrame | EmptyDataFrame:
    """Style EarthRanger features for the map, the same way as STE's get_spatial_features: native EarthRanger
    colours first, then any overrides in `style`. Pass the result to create_spatial_features_layer."""
    if features.empty:
        return cast(EmptyDataFrame, pd.DataFrame())
    server_url = getattr(client, "server", "").rstrip("/")
    layer_style = LayerStyle.model_validate(style[0]) if style else LayerStyle()
    return cast(AnyGeoDataFrame, _apply_geo_style(features, layer_style, group_by, legend_title, server_url))

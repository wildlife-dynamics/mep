from typing import Annotated, Any, TypeAlias, cast

import geopandas as gpd
import pandas as pd
from ecoscope.platform.annotations import (  # type: ignore[import-untyped]
    AdvancedField,
    AnyGeoDataFrame,
    EmptyDataFrame,
)
from ecoscope.platform.connections import (  # type: ignore[import-untyped]
    EarthRangerClient,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self
from wt_registry import register

RGBA: TypeAlias = list[int]  # [R, G, B, A]


class FeatureStyle(BaseModel):
    """Base for per-geometry-type style classes. Provides hex-to-RGBA conversion."""

    @staticmethod
    def _rgba(colors: str | list[str] | list[RGBA]) -> RGBA | list[RGBA]:
        from ecoscope.base.utils import hex_to_rgba  # type: ignore[import-untyped]

        if isinstance(colors, str):
            return list(hex_to_rgba(colors))
        return [c if isinstance(c, list) else list(hex_to_rgba(c)) for c in colors]


class PolygonStyle(FeatureStyle):
    model_config = ConfigDict(title="")
    fill_color: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=[],
            description="Fill hex colour(s) e.g. ['#FFA500']. Cycles across rows.",
        ),
    ] = []
    stroke_color: Annotated[
        str | SkipJsonSchema[None],
        Field(default=None, description="Border hex colour."),
    ] = None
    fill_opacity: Annotated[
        float, Field(default=1.0, ge=0.0, le=1.0, description="Fill opacity 0–1.")
    ] = 1.0
    stroke_width: Annotated[
        float, Field(default=2.0, description="Border width in pixels.")
    ] = 2.0

    @model_validator(mode="after")
    def _convert(self) -> Self:
        if self.fill_color:
            self.fill_color = self._rgba(self.fill_color)  # type: ignore[assignment]
        if isinstance(self.stroke_color, str):
            self.stroke_color = self._rgba(self.stroke_color)  # type: ignore[assignment]
        return self


class LineStyle(FeatureStyle):
    model_config = ConfigDict(title="")
    color: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=[],
            description="Line hex colour(s) e.g. ['#E63946']. Cycles across rows.",
        ),
    ] = []
    opacity: Annotated[
        float, Field(default=1.0, ge=0.0, le=1.0, description="Line opacity 0–1.")
    ] = 1.0
    width: Annotated[float, Field(default=2.0, description="Line width in pixels.")] = (
        2.0
    )

    @model_validator(mode="after")
    def _convert(self) -> Self:
        if self.color:
            self.color = self._rgba(self.color)  # type: ignore[assignment]
        return self


class PointStyle(FeatureStyle):
    model_config = ConfigDict(title="")
    color: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=[],
            description="Fill hex colour(s). For SVG icons this tints the marker. Cycles across rows.",
        ),
    ] = []
    size: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            description="Point radius / icon size in pixels. Leave empty to use the size set in EarthRanger.",
        ),
    ] = None

    @model_validator(mode="after")
    def _convert(self) -> Self:
        if self.color:
            self.color = self._rgba(self.color)  # type: ignore[assignment]
        return self


class LayerStyle(BaseModel):
    model_config = ConfigDict(title="")
    polygon: Annotated[
        list[PolygonStyle],
        Field(
            default=[],
            max_length=1,
            description="Polygon styling. Add one entry to override ER native colours.",
        ),
    ] = []
    line: Annotated[
        list[LineStyle],
        Field(
            default=[],
            max_length=1,
            description="Line styling. Add one entry to override ER native colours.",
        ),
    ] = []
    point: Annotated[
        list[PointStyle],
        Field(
            default=[],
            max_length=1,
            description="Point and icon marker styling. Add one entry to override ER native colours.",
        ),
    ] = []


def _apply_geo_style(
    gdf: AnyGeoDataFrame,
    style: LayerStyle,
    group_by: str = "type_name",
    legend_title: str = "",
    server_url: str = "",
) -> AnyGeoDataFrame:
    gdf = gdf.copy()
    n = len(gdf)

    # Rows with an `image` value are SVG icon markers; all others use geometry fill/stroke.
    is_icon = (
        gdf["image"].notna()
        if "image" in gdf.columns
        else pd.Series(False, index=gdf.index)
    )
    geom_types = gdf.geometry.geom_type
    is_polygon = geom_types.str.contains("Polygon") & ~is_icon
    is_line = geom_types.str.contains("LineString") & ~is_icon
    is_point = ~is_icon & ~is_polygon & ~geom_types.str.contains("LineString")

    def _ensure(col: str) -> None:
        if col not in gdf.columns:
            gdf[col] = [None] * n

    def _set_icon_url() -> None:
        gdf["icon_url"] = None
        gdf.loc[is_icon, "icon_url"] = (server_url + gdf.loc[is_icon, "image"]).values

    if style.polygon or style.line or style.point:
        ps = style.polygon[0] if style.polygon else None
        if ps and is_polygon.any():
            if ps.fill_color:
                alpha = int(ps.fill_opacity * 255)
                _ensure("get_fill_color")
                colors = ps.fill_color
                for i, idx in enumerate(gdf.index[is_polygon]):
                    base = colors[i % len(colors)]
                    gdf.at[idx, "get_fill_color"] = base[:3] + [alpha]  # type: ignore[operator]
            if ps.stroke_color:
                _ensure("get_line_color")
                for idx in gdf.index[is_polygon]:
                    gdf.at[idx, "get_line_color"] = ps.stroke_color
            gdf.loc[is_polygon, "get_line_width"] = ps.stroke_width

        ls = style.line[0] if style.line else None
        if ls and is_line.any():
            if ls.color:
                alpha = int(ls.opacity * 255)
                _ensure("get_line_color")
                colors = ls.color
                for i, idx in enumerate(gdf.index[is_line]):
                    base = colors[i % len(colors)]
                    gdf.at[idx, "get_line_color"] = base[:3] + [alpha]  # type: ignore[operator]
            gdf.loc[is_line, "get_line_width"] = ls.width

        pts = style.point[0] if style.point else None
        if pts:
            if is_point.any():
                if pts.color:
                    _ensure("get_fill_color")
                    colors = pts.color
                    for i, idx in enumerate(gdf.index[is_point]):
                        gdf.at[idx, "get_fill_color"] = colors[i % len(colors)]
                gdf.loc[is_point, "get_point_radius"] = (
                    pts.size if pts.size is not None else 8.0
                )
            if is_icon.any():
                _set_icon_url()
                gdf["icon_size"] = (
                    float(pts.size)
                    if pts.size is not None
                    else pd.to_numeric(
                        gdf.get("width", pd.Series(dtype=float)), errors="coerce"
                    ).fillna(15.0)
                )
                if pts.color:
                    gdf["icon_color"] = None
                    colors = pts.color
                    for i, idx in enumerate(gdf.index[is_icon]):
                        gdf.at[idx, "icon_color"] = colors[i % len(colors)]
    else:
        from ecoscope.base.utils import hex_to_rgba  # type: ignore[import-untyped]

        if "fill" in gdf.columns:
            opacity = gdf.get("fill-opacity", pd.Series(1.0, index=gdf.index))
            gdf["get_fill_color"] = [
                list(hex_to_rgba(str(f))[:3]) + [int(op * 255)]
                if pd.notna(f)
                else [0, 0, 0, 0]
                for f, op in zip(gdf["fill"], opacity)
            ]
            if "stroke" in gdf.columns:
                gdf["get_line_color"] = [
                    list(hex_to_rgba(str(s))) if pd.notna(s) else [0, 0, 0, 0]
                    for s in gdf["stroke"]
                ]
            if "stroke-width" in gdf.columns:
                gdf["get_line_width"] = gdf["stroke-width"].astype(float)

        if (
            "width" in gdf.columns
            and is_point.any()
            and "get_point_radius" not in gdf.columns
        ):
            gdf.loc[is_point, "get_point_radius"] = pd.to_numeric(
                gdf.loc[is_point, "width"], errors="coerce"
            ).fillna(8.0)

        if is_icon.any():
            _set_icon_url()
            gdf["icon_size"] = pd.to_numeric(
                gdf.get("width", pd.Series(dtype=float)), errors="coerce"
            ).fillna(15.0)

    if any(c in gdf.columns for c in ("get_fill_color", "get_line_color", "icon_url")):
        col = (
            group_by
            if (group_by == "geom_type" or group_by in gdf.columns)
            else "geom_type"
        )
        gdf["legend_title"] = legend_title
        gdf["legend_label"] = (
            gdf.geometry.geom_type.astype(str)
            if col == "geom_type"
            else gdf[col].astype(str)
        )

    return gdf


def _featuresets_from_response(
    response: dict[str, Any] | list[Any],
) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        return response.get("features", [])
    return response

class FeatureSetQuery(BaseModel):
    """Load all features from a named EarthRanger featureset."""

    model_config = ConfigDict(title="Feature Set", str_strip_whitespace=True)
    featureset_name: Annotated[
        str,
        Field(
            min_length=1,
            description="Display name of the featureset exactly as it appears in EarthRanger e.g. 'Boundaries'.",
        ),
    ]

    def get(self, client: EarthRangerClient) -> AnyGeoDataFrame:
        response = client._get("featureset/")  # type: ignore[attr-defined]
        featuresets = _featuresets_from_response(response)
        featureset = next(
            (fs for fs in featuresets if fs["name"] == self.featureset_name), None
        )
        if featureset is None:
            raise ValueError(
                f"Featureset {self.featureset_name!r} not found. "
                f"Available: {[fs['name'] for fs in featuresets]}"
            )
        result = get_featureset(client, featureset["id"])
        if not isinstance(result, gpd.GeoDataFrame):
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        return cast(AnyGeoDataFrame, result)


class FeatureTypeQuery(BaseModel):
    """Load all features of a given type across EarthRanger featuresets."""

    model_config = ConfigDict(title="Feature Type", str_strip_whitespace=True)
    feature_type: Annotated[
        str,
        Field(
            min_length=1,
            description="Feature type name as shown in EarthRanger e.g. 'Conservancy'.",
        ),
    ]
    style: Annotated[
        list[LayerStyle],
        Field(
            default=[],
            max_length=1,
            description="Optional: Override how EarthRanger spatial features are rendered on the map. If not specified, features will use their native EarthRanger colours and styling.",
        ),
    ] = []

    def get(self, client: EarthRangerClient) -> AnyGeoDataFrame:
        feature_classes: list[dict[str, Any]] = client._get("featureclass/")  # type: ignore[attr-defined]
        feature_class = next(
            (fc for fc in feature_classes if fc["name"] == self.feature_type), None
        )
        if feature_class is None:
            raise ValueError(f"Feature type {self.feature_type!r} not found.")
        if not feature_class.get("feature_set_id"):
            raise ValueError(
                f"Feature type {self.feature_type!r} is not linked to a featureset."
            )
        result = get_featureset(client, feature_class["feature_set_id"])
        if not isinstance(result, gpd.GeoDataFrame):
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        return cast(
            AnyGeoDataFrame,
            result[result["type_name"] == self.feature_type].reset_index(drop=True),
        )


class FeatureIdQuery(BaseModel):
    """Load a single spatial feature by its EarthRanger UUID."""

    model_config = ConfigDict(title="Feature ID", str_strip_whitespace=True)
    feature_id: Annotated[
        str,
        Field(
            description="UUID of a specific spatial feature available on EarthRanger."
        ),
    ]
    style: Annotated[
        list[LayerStyle],
        Field(
            default=[],
            max_length=1,
            description="Optional: Override how EarthRanger spatial features are rendered on the map. If not specified, features will use their native EarthRanger colours and styling.",
        ),
    ] = []

    def get(self, client: EarthRangerClient) -> AnyGeoDataFrame:
        response = client._get(f"feature/{self.feature_id}/")  # type: ignore[attr-defined]
        if not isinstance(response, dict):
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        if not (features := response.get("features", [])):
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        featuresets = _featuresets_from_response(client._get("featureset/"))  # type: ignore[attr-defined]
        type_map: dict[str, str] = {}
        for featureset in featuresets:
            for entry in featureset.get("types", []):
                type_map[entry["id"]] = entry["name"]
        gdf["type_name"] = gdf["feature_type"].map(type_map)
        return cast(AnyGeoDataFrame, gdf)


EarthRangerQuery: TypeAlias = FeatureSetQuery | FeatureTypeQuery | FeatureIdQuery


class EarthRangerSource(BaseModel):
    model_config = ConfigDict(title="EarthRanger")
    query: Annotated[
        EarthRangerQuery | None,
        Field(default=None),
    ] = None

    def get(self, client: EarthRangerClient) -> AnyGeoDataFrame:
        if self.query is None:
            return cast(AnyGeoDataFrame, gpd.GeoDataFrame())
        return self.query.get(client)


@register()
def get_spatial_features(
    client: EarthRangerClient,
    source: Annotated[
        EarthRangerSource | SkipJsonSchema[None],
        Field(default=None),
    ] = None,
    group_by: Annotated[
        str,
        AdvancedField(
            default="type_name",
            description="Column used to group features in the map legend e.g. 'Feature Type' shows one legend entry per feature type.",
            json_schema_extra={
                "oneOf": [
                    {"const": "type_name", "title": "Feature Type"},
                    {"const": "title", "title": "Feature Name"},
                ]
            },
        ),
    ] = "type_name",
    legend_title: Annotated[
        str,
        AdvancedField(
            default="",
            description="Label shown in the map legend e.g. 'Park Boundary'.",
        ),
    ] = "",
) -> AnyGeoDataFrame | EmptyDataFrame:
    """Load spatial features from EarthRanger and apply styling."""
    if source is None:
        return cast(EmptyDataFrame, pd.DataFrame())
    source_obj = EarthRangerSource.model_validate(source)
    gdf = source_obj.get(client)
    if gdf.empty:
        return cast(EmptyDataFrame, pd.DataFrame())

    query = source_obj.query
    style_list = getattr(query, "style", [])
    server_url = getattr(client, "server", "").rstrip("/")

    style = style_list[0] if style_list else LayerStyle()
    return cast(
        AnyGeoDataFrame,
        _apply_geo_style(gdf, style, group_by, legend_title, server_url),
    )
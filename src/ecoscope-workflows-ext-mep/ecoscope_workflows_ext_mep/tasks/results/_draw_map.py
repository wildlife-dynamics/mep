import pandas as pd
from wt_registry import register
from dataclasses import dataclass
from pydantic.json_schema import SkipJsonSchema
from typing import Annotated, Literal, Tuple, Union, Any
from pydantic import BaseModel, BeforeValidator, Field
from ecoscope_workflows_core.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.results._map import (
    LayerStyleBase,
    ColorAccessor,
    PydeckString,
    LayerDefinition,
    FloatAccessor,
    UnitType,
    TextAnchor,
    AlignmentBaseline,
    LegendDefinition,
    LegendLabel,
    LegendColor,
    _color_tuple_to_css,
    PathLayerStyle,
    ScatterplotLayerStyle,
    PolygonLayerStyle,
    TextLayerStyle,
    IconLayerStyle,
    HexagonLayerStyle,
    GeoJSONLayerStyle,
    view_state_from_layers,
    TiledBitmapLayerDefinition,
    BitmapLayerDefinition,
    LegendStyle,
    ViewState,
    _model_dump_with_pydeck_literals,
    LegendSegment,
    LegendFromDataframe,
    PYDECK_CUSTOM_LIBRARIES,
)

# Published bundle, built+attached to the git tag by .github/workflows/release.yml
# in https://github.com/ttemu34/my-deck-widgets (bump the tag after each release):
PYDECK_CUSTOM_LIBRARIES.append(
    {  # append, don't reassign
        "libraryName": "myDeckWidgets",
        "resourceUri": "https://cdn.jsdelivr.net/gh/ttemu34/my-deck-widgets@v0.5.0/dist/bundle.js",
    }
)


def _camel_case(snake_case: str) -> str:
    head, *tail = snake_case.split("_")
    return head + "".join(word.title() for word in tail)


def _camel_case_nested_dicts(value):
    """
    pydeck camel-cases a layer's own top-level kwarg names, but a nested style
    object (e.g. ``text_style``) is passed through as a plain dict and its keys
    are left as-is. Recurse into dict/list values so nested style dicts read with
    the same camelCase keys as every other prop on the JS side.
    """
    if isinstance(value, dict):
        return {_camel_case(k): _camel_case_nested_dicts(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_camel_case_nested_dicts(v) for v in value]
    return value


@dataclass
class SizeLegendValue:
    """
    One entry in a size-graduated ("bubble") legend: a circle drawn at ``radius``
    screen pixels — the same pixel scale as a ClusteredLabeledScatterplotLayer dot —
    labelled with ``label``.
    """

    label: LegendLabel
    radius: float
    color: LegendColor = "#ff8c00"


@dataclass
class SizeLegendSegment:
    """
    A size-graduated legend: renders one circle per value, sized in pixels, largest
    first, instead of the colour swatches a plain LegendSegment/LegendWidget draws.

    Backed by a custom widget (SizeLegendWidget, in the myDeckWidgets bundle) rather
    than the built-in LegendWidget, since that one only ever reads label+colour.
    """

    values: Annotated[list[SizeLegendValue], Field()]
    title: Annotated[str, Field(default="Legend")] = "Legend"
    # Circles are drawn at min(value.radius, max_radius) — keeps one huge reading
    # from blowing up the legend box. Match this to the layer's cluster_max_radius
    # so the legend's largest circle is the same size as the map's capped dots.
    max_radius: Annotated[float, Field(default=64)] = 64


@dataclass
class SizeLegendFromDataframe:
    """
    A size-graduated legend definition referencing label/radius/colour values in a
    dataframe — analogous to LegendFromDataframe, but building a SizeLegendSegment
    (circles sized by radius) instead of colour swatches.
    """

    title: Annotated[str, AdvancedField(default="Legend")] = "Legend"
    label_column: Annotated[str, AdvancedField(default="labels")] = "labels"
    radius_column: Annotated[str, AdvancedField(default="radius")] = "radius"
    color_column: Annotated[str, AdvancedField(default="colors")] = "colors"
    sort: Annotated[
        Literal["ascending", "descending"] | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None
    label_suffix: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    max_radius: Annotated[float, AdvancedField(default=64)] = 64

    @property
    def display_name(self):
        return self.title.replace("_", " ").title()

    def build_legend_from_dataframe(self, df: AnyGeoDataFrame) -> SizeLegendSegment:
        """
        Lookup the legend label/radius/colour values from the provided dataframe.
        """
        lookup = df.drop_duplicates(subset=self.label_column)[
            [self.label_column, self.radius_column, self.color_column]
        ]
        if self.sort:
            lookup = lookup.sort_values(
                self.radius_column,
                ascending=True if self.sort == "ascending" else False,
            )
        return SizeLegendSegment(
            title=self.display_name,
            max_radius=self.max_radius,
            values=[
                SizeLegendValue(
                    label=f"{row[self.label_column]}{self.label_suffix}"
                    if self.label_suffix
                    else row[self.label_column],
                    radius=row[self.radius_column],
                    color=_color_tuple_to_css(row[self.color_column]),
                )
                for _, row in lookup.iterrows()
            ],
        )


class ClusterLabelStyle(BaseModel):
    """
    Style for the value labels drawn on top of each dot.

    Passed as ``text_style`` to :class:`ClusteredLabeledScatterplotLayerStyle`; set
    ``text_style=None`` there to render only the clustered dots, with no labels at all.
    """

    get_text_color: Annotated[SkipJsonSchema[list[int]] | SkipJsonSchema[None], AdvancedField(default=None)] = None
    text_size: Annotated[float, AdvancedField(default=12)] = 12
    scale_text_to_radius: Annotated[bool, AdvancedField(default=True)] = True
    size_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    font_family: Annotated[PydeckString, AdvancedField(default="Monaco, monospace")] = "Monaco, monospace"
    font_weight: Annotated[PydeckString | SkipJsonSchema[None], AdvancedField(default=None)] = None
    outline_width: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    outline_color: Annotated[
        Tuple[float, float, float, float] | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None
    get_text_anchor: Annotated[TextAnchor, AdvancedField(default="middle")] = "middle"
    get_alignment_baseline: Annotated[AlignmentBaseline, AdvancedField(default="center")] = "center"


class ClusteredLabeledScatterplotLayerStyle(LayerStyleBase):
    """
    Clustered Labeled Scatterplot layer style kwargs.

    A composite layer (registered in the ``ecoscopeDeckWidgets`` custom bundle) that
    draws a ScatterplotLayer with each point's value rendered on top of the dot via a
    TextLayer, and merges points whose (``cluster_max_radius``-capped) dots would touch
    or overlap on screen into a single dot whose label shows the *sum* of the merged
    values. A merged dot's on-screen radius counts toward further merges, so a chain
    of touching dots all collapse into one cluster.

    Because merging is computed in screen space at render time, it is zoom-aware:
    dots merge as you zoom out (where points crowd together) and split back into
    individuals as you zoom in.

    Combines the props of
    https://deck.gl/docs/api-reference/layers/scatterplot-layer and
    https://deck.gl/docs/api-reference/layers/text-layer, plus clustering controls.
    """

    get_position: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    get_value: Annotated[FloatAccessor, AdvancedField(default="value")] = "value"
    aggregate: Annotated[bool, AdvancedField(default=True)] = True
    cluster_padding: Annotated[float, AdvancedField(default=1.0, ge=0)] = 1.0
    cluster_max_radius: Annotated[float, AdvancedField(default=64)] = 64
    get_fill_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_line_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_radius: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=5)] = 5
    get_line_width: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=1)] = 1
    radius_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    radius_scale: Annotated[float, AdvancedField(default=1)] = 1
    radius_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    radius_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    line_width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    line_width_scale: Annotated[float, AdvancedField(default=1)] = 1
    line_width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    line_width_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    stroked: Annotated[bool, AdvancedField(default=True)] = True
    filled: Annotated[bool, AdvancedField(default=True)] = True
    text_style: Annotated[
        ClusterLabelStyle | SkipJsonSchema[None],
        AdvancedField(default=ClusterLabelStyle()),
    ] = ClusterLabelStyle()

class TripsLayerStyle(LayerStyleBase):
    """
    Trips Layer style kwargs
    See https://deck.gl/docs/api-reference/geo-layers/trips-layer for more info
    """

    get_path: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    get_timestamps: Annotated[str, AdvancedField(default="timestamps")] = "timestamps"
    get_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_width: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=1)] = 1
    width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    width_scale: Annotated[float, AdvancedField(default=1)] = 1
    width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    width_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    cap_rounded: Annotated[bool, AdvancedField(default=False)] = False
    joint_rounded: Annotated[bool, AdvancedField(default=False)] = False
    billboard: Annotated[bool, AdvancedField(default=False)] = False
    fade_trail: Annotated[bool, AdvancedField(default=True)] = True
    current_time: Annotated[float, AdvancedField(default=0)] = 0
    trail_length: Annotated[float, AdvancedField(default=0)] = 120


class AnimationWidgetStyle(BaseModel):
    """
    Config for a play/pause + scrub playback widget (AnimationWidget, from the
    myDeckWidgets bundle) that drives a TripsLayer's `current_time` client-side
    via requestAnimationFrame, so the exported map animates instead of only
    ever showing a static snapshot.

    min_time/max_time default to the full span of the first TripsLayer's
    timestamps column found among draw_map's geo_layers, so in the common case
    you only need to set duration_ms/loop/autoplay.
    """

    min_time: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    max_time: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    duration_ms: Annotated[float, AdvancedField(default=15_000, gt=0)] = 15_000
    loop: Annotated[bool, AdvancedField(default=True)] = True
    autoplay: Annotated[bool, AdvancedField(default=True)] = True

    # --- Head-marker tracking (see create_head_marker_layer) --------------------
    head_layer_ids: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Ids of head-marker layers (from create_head_marker_layer) whose "
            "position/orientation AnimationWidget should refresh every tick.",
        ),
    ] = None
    path_field: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    timestamp_field: Annotated[str, AdvancedField(default="times")] = "times"
    min_move_meters: Annotated[
        float,
        AdvancedField(
            default=3.0,
            description="Below this much ground movement (m), hold the last heading "
            "instead of spinning the head marker while a subject is near-stationary.",
        ),
    ] = 3.0

    # --- Auto-rotate camera ------------------------------------------------------
    auto_rotate_speed: Annotated[
        float,
        AdvancedField(
            default=0.0,
            description="Camera rotation speed in degrees/second while playing. "
            "0 = off; positive = clockwise.",
        ),
    ] = 0.0


def _infer_trips_time_bounds(geo_layers: list) -> Tuple[float, float] | None:
    """
    Scan geo_layers for a TripsLayer with a resolved geodataframe, and return
    the (min, max) of its timestamps column — the full span AnimationWidget
    needs to sweep current_time across so playback covers every trip rather
    than whatever an arbitrary default window happens to catch.
    """
    bounds: list[Tuple[float, float]] = []
    for layer_def in geo_layers:
        if layer_def.layer_type != "TripsLayer" or layer_def.geodataframe is None:
            continue
        timestamps_col = getattr(layer_def.layer_style, "get_timestamps", None)
        if not timestamps_col or timestamps_col not in layer_def.geodataframe.columns:
            continue
        flat: list[float] = []
        for value in layer_def.geodataframe[timestamps_col]:
            if isinstance(value, (list, tuple)):
                flat.extend(value)
            else:
                flat.append(value)
        if flat:
            bounds.append((min(flat), max(flat)))
    if not bounds:
        return None
    return min(b[0] for b in bounds), max(b[1] for b in bounds)


class ScenegraphLayerStyle(LayerStyleBase):
    """
    Minimal Scenegraph (3D glTF/GLB model) Layer style kwargs, for the optional
    3D head marker built by create_head_marker_layer.
    See https://deck.gl/docs/api-reference/mesh-layers/scenegraph-layer for more info
    """

    scenegraph: Annotated[str, AdvancedField(default="")] = ""
    get_position: Annotated[str, AdvancedField(default="position")] = "position"
    get_orientation: Annotated[str, AdvancedField(default="orientation")] = "orientation"
    get_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    size_scale: Annotated[float, AdvancedField(default=50.0)] = 50.0
    size_min_pixels: Annotated[float, AdvancedField(default=12.0)] = 12.0
    size_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None


LayerStyle = Union[
    PathLayerStyle,
    ScatterplotLayerStyle,
    PolygonLayerStyle,
    TextLayerStyle,
    IconLayerStyle,
    HexagonLayerStyle,
    GeoJSONLayerStyle,
    TripsLayerStyle,
    ClusteredLabeledScatterplotLayerStyle,
    ScenegraphLayerStyle,
]

@register()
def create_clustered_labeled_scatterplot_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        ClusteredLabeledScatterplotLayerStyle | SkipJsonSchema[None],
        AdvancedField(
            default=ClusteredLabeledScatterplotLayerStyle(),
            description="Style arguments for the layer.",
        ),
    ] = None,
    legend: Annotated[
        LegendDefinition | SizeLegendSegment | SizeLegendFromDataframe | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend. Pass a "
            "SizeLegendSegment or SizeLegendFromDataframe for a size-graduated bubble "
            "legend instead of colour swatches.",
        ),
    ] = None,
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates a clustered labeled scatterplot layer definition.

    Points whose dots would overlap on screen are merged into a single dot whose
    label is the sum of the merged values; clusters split apart as the map zooms in.

    Note: clustering runs client-side over the resolved feature data, so this layer
    expects an in-memory ``geodataframe`` rather than a ``data_url`` (a URL is not
    fetched-and-clustered by the layer).
    """
    return LayerDefinition(
        layer_type="ClusteredLabeledScatterplotLayer",
        layer_style=layer_style or ClusteredLabeledScatterplotLayerStyle(),
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
    )


def _unwrap(v: Any) -> Any:
    if not isinstance(v, list):
        return v
    return [layer for item in v for layer in (item if isinstance(item, list) else [item])]

@register()
def create_trips_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        TripsLayerStyle | SkipJsonSchema[None],
        AdvancedField(default=TripsLayerStyle(), description="Style arguments for the layer."),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates an animated trips layer definition based on the provided configuration.

    If trail_frac and/or current_frac are provided and a geodataframe is given,
    trail_length and current_time are derived from the timeline span of the
    timestamps column referenced by layer_style.get_timestamps.
    """
    return LayerDefinition(
        layer_type="TripsLayer",
        layer_style=layer_style,
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
    )


@register()
def create_head_marker_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="Same geodataframe used for the TripsLayer this marker tracks.", exclude=True),
    ],
    get_color: Annotated[
        ColorAccessor | SkipJsonSchema[None],
        AdvancedField(default=None, description="Column name or literal RGBA for the marker's colour."),
    ] = None,
    radius: Annotated[float, AdvancedField(default=6.0, gt=0)] = 6.0,
    outline_color: Annotated[Tuple[int, int, int], AdvancedField(default=(255, 255, 255))] = (255, 255, 255),
    outline_width: Annotated[float, AdvancedField(default=1.5, ge=0)] = 1.5,
    glb: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Optional glTF/GLB URL for a 3D head model. None -> a flat 2D dot (ScatterplotLayer).",
        ),
    ] = None,
    size_scale: Annotated[float, AdvancedField(default=50.0)] = 50.0,
    size_min_pixels: Annotated[float, AdvancedField(default=12.0)] = 12.0,
    size_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None,
) -> Annotated[LayerDefinition, Field()]:
    """
    Builds the initial (frame-0) head-marker layer from a trips geodataframe: a
    flat dot (ScatterplotLayer) by default, or a 3D glTF model (ScenegraphLayer)
    when glb is given.

    Each row keeps its full LineString geometry + timestamps column alongside a
    derived "position" (and, for the 3D case, "orientation") field. Pass this
    layer's .id into AnimationWidgetStyle.head_layer_ids and AnimationWidget
    refreshes position/orientation every tick straight from this same data —
    no separate sync step, no injected script.
    """
    gdf = geodataframe.copy()
    gdf["position"] = gdf.geometry.apply(lambda g: list(g.coords[0]))
    gdf["orientation"] = [[0.0, 0.0, 0.0] for _ in range(len(gdf))]

    if glb is None:
        style = ScatterplotLayerStyle(
            get_position="position",
            get_fill_color=get_color,
            get_radius=radius,
            radius_units="pixels",
            stroked=outline_width > 0,
            get_line_color=list(outline_color),
            get_line_width=outline_width,
            line_width_units="pixels",
            billboard=True,
        )
        layer_type = "ScatterplotLayer"
    else:
        style = ScenegraphLayerStyle(
            scenegraph=glb,
            get_position="position",
            get_orientation="orientation",
            get_color=get_color,
            size_scale=size_scale,
            size_min_pixels=size_min_pixels,
            size_max_pixels=size_max_pixels,
        )
        layer_type = "ScenegraphLayer"

    return LayerDefinition(
        layer_type=layer_type,
        layer_style=style,
        legend=None,
        geodataframe=gdf,
    )

@register()
def draw_map(
    geo_layers: Annotated[
        LayerDefinition | list[LayerDefinition] | SkipJsonSchema[None],
        BeforeValidator(_unwrap),
        Field(description="A list of map layers to add to the map.", exclude=True),
    ] = None,
    tile_layers: Annotated[
        list[TiledBitmapLayerDefinition | BitmapLayerDefinition] | SkipJsonSchema[None],
        Field(description="A list of tile layers (base maps and/or overlays)."),
    ] = None,
    static: Annotated[bool, Field(description="Set to true to disable map pan/zoom.")] = False,
    title: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default="",
            description="""\
            The map title. Note this is the title drawn on the map canvas itself, and will result
            in duplicate titles if set in the context of a dashboard in which the iframe/widget
            container also has a title set on it.
            """,
        ),
    ] = None,
    legend_style: Annotated[
        LegendStyle | SkipJsonSchema[None],
        AdvancedField(
            default=LegendStyle(),
            description="Additional arguments for configuring the legend.",
        ),
    ] = None,
    max_zoom: Annotated[
        int,
        AdvancedField(
            default=20,
            description="""\
            The maximum zoom level allowed by the map.
            This setting will be overridden if provided
            tile layers max zoom levels are lower than this value.
            """,
        ),
    ] = 20,
    view_state: Annotated[
        ViewState | SkipJsonSchema[None],
        AdvancedField(
            default=ViewState(),
            description="Manually set the view state of the map.",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            The id of the dashboard widget that this tile layer belongs to.
            If set this MUST match the widget title as defined downstream in create_widget tasks
            """,
            exclude=True,
        ),
    ] = None,
    animation: Annotated[
        AnimationWidgetStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="""\
            If present, adds a play/pause + scrub widget that animates a TripsLayer's
            current_time. min_time/max_time are auto-derived from the first TripsLayer
            found in geo_layers when not set explicitly on the AnimationWidgetStyle.
            """,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    """
    Creates a map based on the provided layer definitions and configuration.

    Args:
    geo_layers (LayerDefinition | list[LayerDefinition] | None): Map layers to add to the map.
    tile_layers (list): A named tile layer, ie OpenStreetMap.
    static (bool): Set to true to disable map pan/zoom.
    title (str): The map title.
    legend_style (WidgetStyleBase): Additional arguments for configuring the Legend.
    max_zoom (int): The maximum zoom level of the map
    view_state (ViewState): Manually set the view state of the map, overrides any layer zoom settings.
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks
    animation (AnimationWidgetStyle | None): If present, adds a play/pause + scrub widget
        that animates a TripsLayer's current_time.

    Returns:
    str: A static HTML representation of the map.
    """
    import pydeck as pdk  # type: ignore[import-untyped]

    pdk.settings.custom_libraries = PYDECK_CUSTOM_LIBRARIES

    DEFAULT_WIDGETS = [
        pdk.Widget(
            "NorthArrowWidget",
            placement="top-left",
            id="NorthArrowWidget",
            style={"transform": "scale(0.8)"},
        ),
        pdk.Widget("ScaleWidget", placement="bottom-left", id="ScaleWidget"),
        pdk.Widget("SaveImageWidget", placement="top-right", id="SaveImageWidget"),
    ]

    if tile_layers is None:
        tile_layers = []
    else:
        tile_layers = list(tile_layers)
    if legend_style is None:
        legend_style = LegendStyle()

    legend_values: list = []
    size_legend_segments: list[SizeLegendSegment] = []
    map_layers: list = []
    map_widgets: list = DEFAULT_WIDGETS.copy()

    for tile_layer in tile_layers:
        if isinstance(tile_layer, BitmapLayerDefinition):
            dump = _model_dump_with_pydeck_literals(tile_layer)
            dump.pop("legend", None)
            layer = pdk.Layer("BitmapLayer", **dump)
            map_layers.append(layer)
            if tile_layer.legend is not None:
                legend_values.append(tile_layer.legend)
        else:
            layer = pdk.Layer(
                "TiledBitmapLayer",
                data=tile_layer.url,
                max_zoom=tile_layer.max_zoom,
                min_zoom=tile_layer.min_zoom,
                opacity=tile_layer.opacity,
                tile_size=256,
                widget_id=pdk.types.String(widget_id),
            )
            map_layers.append(layer)
            if tile_layer.max_zoom < max_zoom:
                max_zoom = tile_layer.max_zoom

    # Normalize geo_layers to a list
    if geo_layers is None:
        geo_layers = []
    elif isinstance(geo_layers, LayerDefinition):
        geo_layers = [geo_layers]
    for layer_def in geo_layers:
        # Rendering: prefer data_url if set, fall back to geodataframe
        if layer_def.data_url is not None:
            data = pdk.types.String(layer_def.data_url)
        elif layer_def.geodataframe is not None:
            gdf = layer_def.geodataframe.to_crs("EPSG:4326")  # type: ignore[operator]
            # Pydeck's PolygonLayer does not support MultiPolygon geometries,
            # so we explode them into individual Polygons.
            is_multi = gdf.geometry.geom_type == "MultiPolygon"
            if is_multi.any():
                gdf = pd.concat(
                    [gdf[~is_multi], gdf[is_multi].explode(index_parts=False)],
                    ignore_index=True,
                )
            data = gdf

        style_dump = _model_dump_with_pydeck_literals(layer_def.layer_style)
        style_dump = {k: _camel_case_nested_dicts(v) for k, v in style_dump.items()}
        layer = pdk.Layer(
            type=layer_def.layer_type,
            id=layer_def.id,
            data=data,
            **style_dump,
        )
        map_layers.append(layer)

        # Legend: use geodataframe if present (regardless of rendering path)
        if legend_def := layer_def.legend:
            if isinstance(legend_def, SizeLegendSegment):
                size_legend_segments.append(legend_def)
            elif isinstance(legend_def, SizeLegendFromDataframe):
                if layer_def.geodataframe is not None:
                    size_legend_segments.append(legend_def.build_legend_from_dataframe(layer_def.geodataframe))
                else:
                    print(
                        "SizeLegendFromDataframe legend skipped for layer '%s': "
                        "no geodataframe is available (layer uses data_url). "
                        "Use a SizeLegendSegment to define a static legend for "
                        "URL-backed layers.",
                        layer_def.layer_type,
                    )
            elif isinstance(legend_def, LegendSegment):
                legend_values.append(legend_def)
            elif isinstance(legend_def, LegendFromDataframe):
                if layer_def.geodataframe is not None:
                    legend_values.append(legend_def.build_legend_from_dataframe(layer_def.geodataframe))
                else:
                    print(
                        "LegendFromDataframe legend skipped for layer '%s': "
                        "no geodataframe is available (layer uses data_url). "
                        "Use a LegendSegment to define a static legend for URL-backed layers.",
                        layer_def.layer_type,
                    )

    if legend_values:
        map_widgets.append(
            pdk.Widget(
                "LegendWidget",
                id="LegendWidget",  # TODO remove this once upstream pydeck changes are released
                legend_values=legend_values,
                placement=legend_style.placement,
            )
        )
    for i, segment in enumerate(size_legend_segments):
        map_widgets.append(
            pdk.Widget(
                "SizeLegendWidget",
                id=f"SizeLegendWidget-{i}",
                title=segment.title,
                max_radius=segment.max_radius,
                values=[{"label": v.label, "radius": v.radius, "color": v.color} for v in segment.values],
                placement=legend_style.placement,
            )
        )

    if title:
        map_widgets.append(
            pdk.Widget(
                "TitleWidget",
                id="TitleWidget",  # TODO remove this once upstream pydeck changes are released
                title=title,
            )
        )

    if animation is not None:
        min_time, max_time = animation.min_time, animation.max_time
        if min_time is None or max_time is None:
            inferred = _infer_trips_time_bounds(geo_layers)
            if inferred is None:
                raise ValueError(
                    "animation requires min_time/max_time: no TripsLayer with a "
                    "resolved geodataframe and timestamps column was found in "
                    "geo_layers to infer them from. Pass them explicitly on "
                    "AnimationWidgetStyle instead."
                )
            min_time = min_time if min_time is not None else inferred[0]
            max_time = max_time if max_time is not None else inferred[1]
        map_widgets.append(
            pdk.Widget(
                "AnimationWidget",
                id="AnimationWidget",
                min_time=min_time,
                max_time=max_time,
                duration_ms=animation.duration_ms,
                loop=animation.loop,
                autoplay=animation.autoplay,
                head_layer_ids=animation.head_layer_ids,
                path_field=animation.path_field,
                timestamp_field=animation.timestamp_field,
                min_move_meters=animation.min_move_meters,
                auto_rotate_speed=animation.auto_rotate_speed,
            )
        )

    m = pdk.Deck(
        layers=map_layers,
        widgets=map_widgets,
        initial_view_state=view_state or view_state_from_layers(layers=geo_layers, max_zoom=max_zoom),
        views=pdk.View(
            "MapView",
            controller=not static,
            repeat=True,
        ),
        parameters={"depthTest": any([getattr(layer, "extruded", False) for layer in map_layers])},
        map_style=pdk.map_styles.LIGHT_NO_LABELS,
    )

    return m.to_html(as_string=True)




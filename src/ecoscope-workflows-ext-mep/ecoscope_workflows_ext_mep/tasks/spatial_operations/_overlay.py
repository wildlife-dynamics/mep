from wt_registry import register
from typing import Literal, cast
from ecoscope.platform.annotations import AnyGeoDataFrame

@register()
def overlay_gdf(
    left: AnyGeoDataFrame,
    right: AnyGeoDataFrame,
    how: Literal[
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",
    ] = "intersection",
    keep_geom_type: bool | None = None,
    make_valid: bool = True,
) -> AnyGeoDataFrame:
    """Overlay two GeoDataFrames.

    Parameters
    ----------
    left : GeoDataFrame
        The left frame (the one `.overlay` is called on).
    right : GeoDataFrame
        The frame overlaid against `left`.
    how : {"intersection", "union", "identity",
           "symmetric_difference", "difference"}, default "intersection"
        Overlay operation.
    keep_geom_type : bool or None, default None
        If True, return only geometries of the same type as the inputs.
    make_valid : bool, default True
        Attempt to make invalid geometries valid before overlaying.

    Returns
    -------
    GeoDataFrame
    """
    result = left.overlay(
        right,
        how=how,
        keep_geom_type=keep_geom_type,
        make_valid=make_valid,
    )
    return cast(AnyGeoDataFrame, result)


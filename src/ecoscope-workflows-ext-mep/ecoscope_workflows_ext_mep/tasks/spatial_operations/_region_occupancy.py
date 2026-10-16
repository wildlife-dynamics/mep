from typing import Annotated, cast

import geopandas as gpd
import pandas as pd
from ecoscope.platform.annotations import AdvancedField, AnyDataFrame, AnyGeoDataFrame
from pydantic import Field
from shapely.geometry import Polygon
from wt_registry import register

# Columns tried, in order, for region names when no `region_column` is given.
REGION_NAME_COLUMNS = ("name", "region", "title", "short_name", "id", "globalid")


def _has_area(geometries) -> "pd.Series":
    """True where a geometry is present and not empty."""
    return geometries.map(lambda g: g is not None and not g.is_empty).astype(bool)


def _region_names(regions_gdf, region_column: str | None) -> list[str]:
    """A name for each region: from `region_column`, else the first name-like column, else "Region <n>".
    Blank names also fall back to "Region <n>", so unnamed rows are never merged together."""
    if not region_column:
        region_column = next((c for c in REGION_NAME_COLUMNS if c in regions_gdf.columns), None)
    names = regions_gdf[region_column] if region_column else [None] * len(regions_gdf)
    return [
        str(name).strip() if pd.notna(name) and str(name).strip() else f"Region {i}"
        for i, name in enumerate(names, start=1)
    ]


@register()
def compute_region_occupancy(
    etd_gdf: Annotated[
        AnyGeoDataFrame,
        Field(
            description="Time density ranges, with a `percentile` column, e.g. from calculate_elliptical_time_density."
        ),
    ],
    regions_gdf: Annotated[
        AnyGeoDataFrame, Field(description="Regions to measure occupancy in, one or more rows each.")
    ],
    region_column: Annotated[
        str | None,
        Field(
            description="Column in `regions_gdf` naming each region; rows sharing a name are merged. "
            f"Leave empty to use the first of {', '.join(REGION_NAME_COLUMNS)} that exists, "
            "or number the regions if none does."
        ),
    ] = None,
    percentile: Annotated[
        float, AdvancedField(default=99.9, description="Which time density percentile to use as the home range.")
    ] = 99.9,
    subject_column: Annotated[
        str,
        AdvancedField(
            default="subject_name",
            description="Column in `etd_gdf` naming each subject; if it is missing, all ranges are one subject.",
        ),
    ] = "subject_name",
) -> AnyDataFrame:
    """The share of each subject's home range (its `percentile` time density range) that falls in each region.

    Returns one row per subject and region: `region`, `region_area_sqkm` (the whole region; empty for the
    outside row), `area_sqkm` (home range inside the region), `home_range_sqkm` and `occupancy_pct`.
    Regions may overlap, so their percentages can add up to more than 100. The "Outside regions" row is
    the part of the range in no region at all, so it never double counts. Areas are measured in meters
    in the local UTM zone of the ranges.
    """
    from shapely import make_valid

    ranges = etd_gdf[etd_gdf["percentile"] == percentile]
    if ranges.empty:
        raise ValueError(
            f"`compute_region_occupancy`: no {percentile} percentile in etd_gdf. "
            f"Available percentiles: {sorted(etd_gdf['percentile'].unique().tolist())}"
        )
    ranges = ranges[_has_area(ranges.geometry)]
    if ranges.empty:
        raise ValueError(f"`compute_region_occupancy`: the {percentile} percentile home range has no area.")
    # UTM is in meters and keeps areas true within its zone; Web Mercator would inflate them away from the equator.
    crs = ranges.estimate_utm_crs()
    ranges = ranges.to_crs(crs)

    regions = gpd.GeoDataFrame(
        {"region": _region_names(regions_gdf, region_column)}, geometry=regions_gdf.geometry.values, crs=regions_gdf.crs
    ).to_crs(crs)
    regions = regions[_has_area(regions.geometry)]
    regions = regions.assign(geometry=regions.geometry.map(make_valid)).dissolve(by="region", sort=False)
    by_subject = ranges.groupby(subject_column, sort=False) if subject_column in ranges.columns else [(None, ranges)]

    rows = []
    for subject, subject_ranges in by_subject:
        home_range = make_valid(subject_ranges.union_all())
        total = home_range.area
        if home_range.is_empty or total == 0:
            raise ValueError(f"`compute_region_occupancy`: home range has no area for subject '{subject}'.")
        overlaps = [(name, geom.area, geom.intersection(home_range).area) for name, geom in regions.geometry.items()]
        for region, region_area, area in overlaps:
            row = {} if subject is None else {subject_column: subject}
            rows.append(
                {
                    **row,
                    "region": region,
                    "region_area_sqkm": None if region_area is None else round(region_area / 1e6, 3),
                    "area_sqkm": round(area / 1e6, 3),
                    "occupancy_pct": round(100 * area / region_area, 1),
                }
            )
    return cast(AnyDataFrame, pd.DataFrame(rows))

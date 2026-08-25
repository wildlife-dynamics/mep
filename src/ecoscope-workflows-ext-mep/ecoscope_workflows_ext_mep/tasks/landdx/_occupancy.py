from shapely.geometry import Polygon, MultiPolygon
from typing import cast,Dict,Union,List
import pandas as pd 
from wt_registry import register
from ecoscope.platform.annotations import AnyGeoDataFrame,AnyDataFrame

@register()
def build_ldx_template_region_lookup(
    gdf: AnyGeoDataFrame,
) -> Dict[str, List[str]]:
    """
    Build a lookup dictionary grouping region IDs by category type.
    """
    gdf = gdf.reset_index()
    categories ={
        "national_pa_use": [
            "National Park",
            "National Reserve",
            "National Reserve_Privately Managed",
            "National Reserve_Beacon_Adjusted",
            "Forest Reserve",
        ],
        "community_pa_use": ["Community Conservancy", "Group Ranch"],
    }

    # Default static UUIDs
    static_ids ={
        "crop_raid_percent": ["2d3f6392-700c-495f-8bc5-087538f6f125"],
        "kenya_use": ["7895ded1-df29-4ca1-8e34-ebc8e3cbb24e"],
    }
    result = {}
    for name, types in categories.items():
        mask = gdf["type"].isin(types) & gdf["type"].notna() & ~gdf.is_empty
        result[name] = gdf.loc[mask, "globalid"].dropna().tolist()
    result.update(static_ids)
    return result

@register()
def compute_template_regions(
    geodataframe: AnyGeoDataFrame, 
    template_lookup: Dict[str, list[str]], 
    crs: str
) -> Dict[str, Polygon | MultiPolygon]:
    return {
        template: geodataframe.query("globalid in @gids").to_crs(crs).union_all()
        for template, gids in template_lookup.items()
    }

@register() 
def compute_subject_occupancy(
    subjects_df: AnyDataFrame,
    crs: str,
    etd_gdf: AnyGeoDataFrame,
    regions_gdf: Dict[str, Union[Polygon, MultiPolygon]],
) -> AnyDataFrame:
    subject_id = subjects_df["subject_name"].iloc[0]
    # Get home range at 99.9th percentile and convert to target CRS
    try:
        percentile_mask = etd_gdf["percentile"] == 99.999
        if not percentile_mask.any():
            raise ValueError(
                f"`compute_subject_occupancy`:No 99.9th percentile found for subject '{subject_id}'. "
                f"Available percentiles: {sorted(etd_gdf['percentile'].unique().tolist())}"
            )

        subject_range = etd_gdf[percentile_mask].to_crs(crs).geometry.iloc[0]

    except (IndexError, KeyError) as e:
        raise ValueError(
            f"`compute_subject_occupancy`:Could not extract 99.9th percentile ETD for subject '{subject_id}'. "
            f"Available percentiles: {etd_gdf['percentile'].unique().tolist()}"
        ) from e

    if subject_range.is_empty:
        raise ValueError(f"`compute_subject_occupancy`:Home range geometry is empty for subject '{subject_id}'.")
    total_area = subject_range.area
    if total_area == 0:
        raise ValueError(f"`compute_subject_occupancy`:Home range has zero area for subject '{subject_id}'.")
    occupancy = {}
    for region_name, region_geom in regions_gdf.items():
        try:
            intersection_area = region_geom.intersection(subject_range).area
            occupancy[region_name] = 100 * (intersection_area / total_area)
        except Exception as e:
            print(f"Warning: Failed to compute intersection for region '{region_name}': {e}")
            occupancy[region_name] = 0.0

    # Calculate unprotected area
    national_pa = occupancy.get("national_pa_use", 0.0)
    community_pa = occupancy.get("community_pa_use", 0.0)
    occupancy["unprotected"] = max(0.0, 100.0 - national_pa - community_pa)
    occupancy = {k: round(v, 1) for k, v in occupancy.items()}
    return cast(AnyDataFrame, pd.DataFrame([occupancy]))

import os
import logging
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from ecoscope_workflows_core.decorators import task
from typing import Union, cast, Optional, Dict, List
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

logger = logging.getLogger(__name__)

@task
def build_template_region_lookup(
    gdf: AnyGeoDataFrame,
    categories: Optional[Dict[str, List[str]]] = None,
    static_ids: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Build a lookup dictionary grouping region IDs by category type.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`build_template_region_lookup`:gdf cannot be empty.")

    gdf = gdf.reset_index()

    # Default categories (you can add more later)
    categories = categories or {
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
    static_ids = static_ids or {
        "crop_raid_percent": ["2d3f6392-700c-495f-8bc5-087538f6f125"],
        "kenya_use": ["7895ded1-df29-4ca1-8e34-ebc8e3cbb24e"],
    }
    result = {}
    for name, types in categories.items():
        mask = gdf["type"].isin(types) & gdf["type"].notna() & ~gdf.is_empty
        result[name] = gdf.loc[mask, "globalid"].dropna().tolist()

    result.update(static_ids)
    return result


@task
def compute_template_regions(
    geodataframe: AnyGeoDataFrame, template_lookup: Dict[str, list[str]], crs: str
) -> Dict[str, Polygon | MultiPolygon]:
    return {
        template: geodataframe.query("globalid in @gids").to_crs(crs).unary_union
        for template, gids in template_lookup.items()
    }


@task
def compute_subject_occupancy(
    subjects_df: AnyDataFrame,
    crs: str,
    etd_gdf: AnyGeoDataFrame,
    regions_gdf: Dict[str, Union[Polygon, MultiPolygon]],
    output_path: Union[str, Path],
) -> AnyDataFrame:
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = remove_file_scheme(output_path)
    if subjects_df is None or subjects_df.empty:
        raise ValueError("`compute_subject_occupancy`:Subjects dataframe is empty.")
    if etd_gdf is None or etd_gdf.empty:
        raise ValueError("`compute_subject_occupancy`:ETD GeoDataFrame is empty.")
    if regions_gdf is None or not regions_gdf:
        raise ValueError("`compute_subject_occupancy`:Regions dictionary is empty.")

    subject_id = subjects_df["subject_name"].iloc[0]

    # Get home range at 99.9th percentile and convert to target CRS
    try:
        percentile_mask = etd_gdf["percentile"] == 99.9
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
            logger.error(f"Warning: Failed to compute intersection for region '{region_name}': {e}")
            occupancy[region_name] = 0.0

    # Calculate unprotected area
    national_pa = occupancy.get("national_pa_use", 0.0)
    community_pa = occupancy.get("community_pa_use", 0.0)
    occupancy["unprotected"] = max(0.0, 100.0 - national_pa - community_pa)
    occupancy = {k: round(v, 1) for k, v in occupancy.items()}
    return cast(AnyDataFrame, pd.DataFrame([occupancy]))

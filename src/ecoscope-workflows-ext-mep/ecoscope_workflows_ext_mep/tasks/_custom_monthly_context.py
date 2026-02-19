import os
import uuid
import logging 
import pandas as pd
from pathlib import Path
from datetime import datetime
from docx.shared import Inches
from docxtpl import InlineImage
from docxtpl import DocxTemplate
from typing import Optional, Dict, Any,List
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.skip import SKIP_SENTINEL, SkipSentinel
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

logger = logging.getLogger(__name__)

def validate_image_path(field_name: str, path: str) -> None:
    """Validate that an image file exists and has valid extension."""
    normalized_path = remove_file_scheme(path)

    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"Image file for '{field_name}' not found: {normalized_path}")

    valid_extensions = {".png", ".jpg", ".jpeg"}
    if Path(normalized_path).suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Invalid image format for '{field_name}': {Path(normalized_path).suffix}. "
            f"Expected one of {valid_extensions}"
        )

    logger.info(f" Validated image for '{field_name}': {normalized_path}")
    
def _unwrap_and_validate_list(
    paths: List[str | SkipSentinel | None] | None,
) -> List[str]:
    """
    Unwrap sentinels, drop None values, drop non-existent paths.
    Returns a clean list of validated absolute path strings.
    """
    if not paths:
        return []
    validated: List[str] = []
    for i, p in enumerate(paths):
        p = _unwrap_skip(p)
        if p is not None:
            validated.append(p)
    return validated

def _unwrap_skip(value):
    """
    Unwrap SkipSentinel values, converting them to None.
    """
    if value is None or value is SKIP_SENTINEL:
        return None
    return value

def _stem_to_label(path: str) -> str:
    """
    Derive a human-readable label from a file-path stem.

    Examples
    --------
    "tembo.png"            → "Tembo"
    "tsavo_east.png"       → "Tsavo East"
    "/data/ambo bull.png"  → "Ambo Bull"
    """
    stem = Path(path).stem   
    return stem.replace("_", " ").replace("-", " ").title()

def safe_read_csv(file_path: str | None) -> pd.DataFrame:
    """
    Safely read CSV file and return DataFrame.

    Args:
        file_path: Path to CSV file or None

    Returns:
        DataFrame with data, or empty DataFrame if file is invalid
    """
    if file_path is None:
        logger.warning("CSV file path is None")
        return pd.DataFrame()

    if not file_path.strip():
        logger.warning("CSV file path is empty string")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"CSV file is empty: {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty or corrupted: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return pd.DataFrame()

@task
def create_monthly_ctx_cover(
    report_period: TimeRange,
    prepared_by: str,
) -> Dict[str, str]:
    """
    Build a dictionary with the mapbook report template values.

    Args:
        count (int): Total number of subjects or records.
        report_period (TimeRange): Object with 'since', 'until', and 'time_format' attributes.
        prepared_by (str): Name of the person or organization preparing the report.

    Returns:
        Dict[str, str]: Structured dictionary with formatted metadata.
    """

    formatted_date = datetime.now()
    formatted_date_str = formatted_date.strftime("%Y-%m-%d %H:%M:%S")
    fmt = getattr(report_period, "time_format", "%Y-%m-%d")
    formatted_time_range = f"{report_period.since.strftime(fmt)} to {report_period.until.strftime(fmt)}"

    # Return structured dictionary
    return {
        "report_id": f"REP-{uuid.uuid4().hex[:8].upper()}",
        "time_generated": formatted_date_str,
        "report_period": formatted_time_range,
        "prepared_by": prepared_by,
    }
     
@task   
def create_mep_monthly_context(
    elephant_sightings_map_path: str | SkipSentinel | None,
    speedmap_path: str | SkipSentinel | None,
    foot_patrols_map_path: str | SkipSentinel | None,
    vehicle_patrol_map_path: str | SkipSentinel | None,
    collared_elephant_plot_paths: List[str | SkipSentinel | None] | None,
    regional_ndvi_plot_paths: List[str | SkipSentinel | None] | None,
    stirep_df_path: str | SkipSentinel | None,
    template_path: str,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    template_path = remove_file_scheme(template_path)
    output_dir    = remove_file_scheme(output_dir)

    speedmap_path               = _unwrap_skip(speedmap_path)
    elephant_sightings_map_path = _unwrap_skip(elephant_sightings_map_path)
    foot_patrols_map_path       = _unwrap_skip(foot_patrols_map_path)
    vehicle_patrol_map_path     = _unwrap_skip(vehicle_patrol_map_path)
    stirep_df_path              = _unwrap_skip(stirep_df_path)  # fix: was sitrep_df_path

    collared_paths = _unwrap_and_validate_list(collared_elephant_plot_paths)
    ndvi_paths     = _unwrap_and_validate_list(regional_ndvi_plot_paths)

    if not filename:
        filename = f"mep_report_{uuid.uuid4().hex[:4]}.docx"
    output_path = Path(output_dir) / filename

    try:
        tpl = DocxTemplate(template_path)
        logger.info(f"Loaded template: {template_path}")
    except Exception as e:
        raise ValueError(f"Failed to load template: {e}")
    
    collar_voltage_list: List[Dict[str, Any]] = [
        {
            "collar_voltage_image": InlineImage(tpl, path, width=Inches(6.58), height=Inches(3.85)),
            "subject":             _stem_to_label(path),
        }
        for path in collared_paths
    ]

    ndvi_list: List[Dict[str, Any]] = [
        {
            "ndvi_image":  InlineImage(tpl, path, width=Inches(6.58), height=Inches(3.85)),
            "area":       _stem_to_label(path),
        }
        for path in ndvi_paths
    ]
    
    sitrep_df = safe_read_csv(stirep_df_path) 
    sitrep    = sitrep_df.to_dict(orient="records")

    context = {
        "elephant_speedmap":     InlineImage(tpl, speedmap_path, width=Inches(6.58), height=Inches(3.85)),
        "elephant_sighting_map": InlineImage(tpl, elephant_sightings_map_path, width=Inches(6.58), height=Inches(3.85)),
        "vehicle_patrol_tracks": InlineImage(tpl, vehicle_patrol_map_path, width=Inches(6.58), height=Inches(3.85)),
        "foot_patrol_tracks":    InlineImage(tpl, foot_patrols_map_path, width=Inches(6.58), height=Inches(3.85)),
        "sitrep":                sitrep,
        "collar_voltage_list":   collar_voltage_list,
        "ndvi_list":             ndvi_list,
    }
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        tpl.render(context)
        tpl.save(output_path)
        logger.info(f"Saved document to: {output_path}")
        return str(output_path)
    except Exception as e:
        raise ValueError(f"Failed to render or save document: {e}")
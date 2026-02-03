import os
import uuid
import logging
import warnings
import pandas as pd
from pathlib import Path
from pydantic import Field
from datetime import datetime
from docxtpl import DocxTemplate
from ecoscope_workflows_core.decorators import task
from typing import Annotated, Optional, Dict, Any
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
from docxtpl import InlineImage
from docx.shared import Inches

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@task
def create__mep_context_page(
    template_path: Annotated[
        str,
        Field(
            description="Path to the .docx template file.",
        ),
    ],
    output_dir: Annotated[
        str,
        Field(
            description="Directory to save the generated .docx file.",
        ),
    ],
    context: Annotated[
        dict,
        Field(
            description="Dictionary with context values for the template.",
        ),
    ],
    filename: Annotated[
        Optional[str],
        Field(
            description="Optional filename . If not provided, a random UUID-based filename will be generated.",
            exclude=True,
        ),
    ] = None,
) -> Annotated[
    str,
    Field(
        description="Full path to the generated .docx file.",
    ),
]:
    """
    Create a context page document from a template and context dictionary.

    Args:
        template_path (str): Path to the .docx template file.
        output_dir (str): Directory to save the generated .docx file.
        context (dict): Dictionary with context values for the template.
        filename (str, optional): Optional filename for the generated file.
            If not provided, a random UUID-based filename will be generated.

    Returns:
        str: Full path to the generated .docx file.
    """
    # Normalize paths
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    # Validate paths
    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = "context_page_.docx"
    output_path = Path(output_dir) / filename

    doc = DocxTemplate(template_path)
    doc.render(context)
    doc.save(output_path)
    return str(output_path)


@task
def create_mep_ctx_cover(
    count: int,
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
        "subject_count": str(count),
        "time_generated": formatted_date_str,
        "report_period": formatted_time_range,
        "prepared_by": prepared_by,
    }


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

    print(f" Validated image for '{field_name}': {normalized_path}")


# subject template
# 1. profile photo  -- download_profile_pic
# 2. subject information -- download_subject_info
# 3. speedmap  -- convert_speedmap_png
# 4. homerange map  -- convert_homerange_png
# 5. seasonal homerange map -- convert_season_png
# 6. nsd plot  -- convert_nsd_png
# 7. speed plot -- convert_speed_png
# 8. collared event plot -- convert_events_png
# 9. mcp plot -- convert_mcp_png
# 10. subject stats table -- persist_subject_stats
# 11. subject occupancy table -- persist_subject_occupancy


@task
def create_mep_subject_context(
    profile_photo_path: str | None,
    subject_info_path: str | None,
    speedmap_path: str | None,
    homerange_map_path: str | None,
    seasonal_homerange_map_path: str | None,
    nsd_plot_path: str | None,
    speed_plot_path: str | None,
    collared_event_plot_path: str | None,
    mcp_plot_path: str | None,
    subject_stats_table_path: str | None,
    subject_occupancy_table_path: str | None,
) -> Dict[str, Any]:
    """
    Build a dictionary with the subject report template values.

    Handles None values gracefully for all input parameters.

    Args:
        profile_photo_path: Path to the profile photo or None.
        subject_info_path: Path to the subject information CSV or None.
        speedmap_path: Path to the speedmap image or None.
        homerange_map_path: Path to the homerange map image or None.
        seasonal_homerange_map_path: Path to the seasonal homerange map image or None.
        nsd_plot_path: Path to the NSD plot image or None.
        speed_plot_path: Path to the speed plot image or None.
        collared_event_plot_path: Path to the collared event plot image or None.
        mcp_plot_path: Path to the MCP plot image or None.
        subject_stats_table_path: Path to the subject stats CSV or None.
        subject_occupancy_table_path: Path to the subject occupancy CSV or None.

    Returns:
        Structured dictionary with subject report values. Missing values default to appropriate fallbacks.
    """

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

    def safe_get_value(df: pd.DataFrame, column: str, default: Any = None, row: int = 0) -> Any:
        """
        Safely extract value from DataFrame with fallback.

        Args:
            df: DataFrame to extract from
            column: Column name
            default: Default value if extraction fails
            row: Row index (default: 0)

        Returns:
            Extracted value or default
        """
        if df.empty or column not in df.columns or len(df) <= row:
            return default

        value = df.iloc[row][column]
        return value if pd.notna(value) else default

    def validate_path(path: str | None, path_name: str) -> str | None:
        """
        Validate and log path status.

        Args:
            path: Path to validate or None
            path_name: Human-readable name for logging

        Returns:
            Original path if valid, None otherwise
        """
        if path is None:
            logger.warning(f"{path_name} is None")
            return None

        if not path.strip():
            logger.warning(f"{path_name} is empty string")
            return None

        from pathlib import Path

        if not Path(path).exists():
            logger.warning(f"{path_name} does not exist: {path}")
            return None

        return path

    # Validate and log all image/file paths
    profile_photo_path = validate_path(profile_photo_path, "Profile photo")
    speedmap_path = validate_path(speedmap_path, "Speedmap")
    homerange_map_path = validate_path(homerange_map_path, "Homerange map")
    seasonal_homerange_map_path = validate_path(seasonal_homerange_map_path, "Seasonal homerange map")
    nsd_plot_path = validate_path(nsd_plot_path, "NSD plot")
    speed_plot_path = validate_path(speed_plot_path, "Speed plot")
    collared_event_plot_path = validate_path(collared_event_plot_path, "Collared event plot")
    mcp_plot_path = validate_path(mcp_plot_path, "MCP plot")

    # Read data files
    subject_stats_df = safe_read_csv(subject_stats_table_path)
    subject_info_df = safe_read_csv(subject_info_path)
    subject_occupancy_df = safe_read_csv(subject_occupancy_table_path)

    # Extract subject stats with defaults
    name = safe_get_value(subject_stats_df, "name", None)
    if not name or name == "Unknown":  # Handles None, empty string, and "Unknown"
        name = safe_get_value(subject_info_df, "subject_name", "Unknown")

    mcp = safe_get_value(subject_stats_df, "MCP", 0.0)
    etd = safe_get_value(subject_stats_df, "ETD", 0.0)
    time_tracked_days = safe_get_value(subject_stats_df, "time_tracked_days", 0)
    time_tracked_years = safe_get_value(subject_stats_df, "time_tracked_years", 0.0)
    distance_travelled = safe_get_value(subject_stats_df, "distance_travelled", 0.0)
    max_displacement = safe_get_value(subject_stats_df, "max_displacement", 0.0)
    night_day_ratio = safe_get_value(subject_stats_df, "night_day_ratio", 0.0)

    # Extract subject info with defaults
    dob_raw = safe_get_value(subject_info_df, "dob", None)
    dob = str(int(dob_raw)) if dob_raw is not None and pd.notna(dob_raw) else "-"
    sex = safe_get_value(subject_info_df, "sex", "-")
    country = safe_get_value(subject_info_df, "country", "-")
    notes = safe_get_value(subject_info_df, "notes", "None")
    status = safe_get_value(subject_info_df, "status_raw", "-")
    bio = safe_get_value(subject_info_df, "bio", "")
    distribution = safe_get_value(subject_info_df, "distribution", "")

    # Extract occupancy data with defaults
    national_pa_use = safe_get_value(subject_occupancy_df, "national_pa_use", 0.0)
    community_pa_use = safe_get_value(subject_occupancy_df, "community_pa_use", 0.0)
    crop_raid_percent = safe_get_value(subject_occupancy_df, "crop_raid_percent", 0.0)
    kenya_use = safe_get_value(subject_occupancy_df, "kenya_use", 0.0)
    unprotected = safe_get_value(subject_occupancy_df, "unprotected", 0.0)

    # Build context dictionary (None values are allowed and will be handled by template)
    ctx = {
        # Media paths (can be None if files don't exist)
        "profile_photo": profile_photo_path,
        "mov_map": speedmap_path,
        "overview_map": homerange_map_path,
        "range_map": seasonal_homerange_map_path,
        "nsd_plot": nsd_plot_path,
        "speed_plot": speed_plot_path,
        "collar_event_timeline": collared_event_plot_path,
        "mcp_plot": mcp_plot_path,
        # Subject statistics
        "name": name,
        "mcp": mcp,
        "etd": etd,
        "time_tracked_days": time_tracked_days,
        "time_tracked_years": time_tracked_years,
        "distance_travelled": distance_travelled,
        "max_displacement": max_displacement,
        "night_day_ratio": night_day_ratio,
        "distribution": distribution,
        # Subject information
        "dob": dob,
        "sex": sex,
        "country": country,
        "id_notes": notes,
        "status": status,
        "bio": bio,
        # Occupancy data
        "national_pa_use": national_pa_use,
        "community_pa_use": community_pa_use,
        "crop_raid_percent": crop_raid_percent,
        "kenya_use": kenya_use,
        "unprotected": unprotected,
    }

    # Count how many paths are None
    none_paths = sum(1 for k, v in ctx.items() if k.endswith(("_photo", "_map", "_plot", "_timeline")) and v is None)
    total_paths = 8

    logger.info(f"Created context for subject: {name}")
    logger.info(f"Media files available: {total_paths - none_paths}/{total_paths}")
    logger.debug(f"Full context: {ctx}")
    print(f"Full context: {ctx}")
    print(f" Created context for subject: {name}")
    print(f" Media files available: {total_paths - none_paths}/{total_paths}")

    return ctx


# Define image dimension configurations (in cm)
IMAGE_DIMENSIONS = {
    # Wide timeline/plot images (2238x450)
    "timeline_plot": {"width": 15.0, "height": 3.0},  # Maintains aspect ratio ~5:1
    # Tall range map (602x855)
    "range_map": {"width": 8.0, "height": 11.4},  # Maintains aspect ratio ~0.7:1
    # Medium landscape maps (765x525)
    "landscape_map": {"width": 12.0, "height": 8.2},  # Maintains aspect ratio ~1.45:1
    # Profile photo (square/portrait)
    "profile_photo": {"width": 3.75, "height": 3.75},  # Adjust as needed
}


def create_inline_image_inch(template: DocxTemplate, image_path: str, width_cm: float, height_cm: float) -> InlineImage:
    """
    Create an InlineImage object with specified dimensions.

    Args:
        template: DocxTemplate instance
        image_path: Path to the image file
        width_cm: Width in centimeters
        height_cm: Height in centimeters

    Returns:
        InlineImage object ready for template rendering
    """
    return InlineImage(template, image_path, width=Inches(width_cm), height=Inches(height_cm))


def prepare_mep_context_for_template(
    context: Dict[str, Any],
    template: DocxTemplate,
) -> Dict[str, Any]:
    """
    Prepare context by converting image paths to InlineImage objects.

    Args:
        context: Original context dictionary
        template: DocxTemplate instance

    Returns:
        Updated context with InlineImage objects
    """
    # Define which fields map to which image dimensions
    image_field_mapping = {
        # Timeline/plot images (wide format)
        "collar_event_timeline": {"height": 1.58, "width": 10.54},
        "nsd_plot": {"height": 2.61, "width": 10.58},
        "speed_plot": {"height": 2.61, "width": 10.58},
        "mcp_plot": {"height": 2.61, "width": 10.58},
        # Range map -- seasons
        "range_map": {"height": 7.08, "width": 5.36},
        # mov_map --speedmap
        "mov_map": {"height": 4.42, "width": 7.47},
        # overview map -- home range
        "overview_map": {"height": 7.06, "width": 5.4},
        # Profile photo
        "profile_photo": {"height": 3.58, "width": 3.44},
    }

    rendered_context = context.copy()

    for field_name, dimensions in image_field_mapping.items():
        if field_name in rendered_context:
            image_path = rendered_context[field_name]

            # Skip if path is None or empty
            if not image_path:
                logger.warning(f"Empty image path for field: {field_name}")
                continue

            # Verify file exists
            if not Path(image_path).exists():
                logger.error(f"Image file not found for {field_name}: {image_path}")
                continue

            try:
                rendered_context[field_name] = create_inline_image_inch(
                    template=template,
                    image_path=image_path,
                    width_cm=dimensions["width"],
                    height_cm=dimensions["height"],
                )
                logger.debug(f"Created InlineImage for {field_name}: {dimensions['width']}x{dimensions['height']} cm")
            except Exception as e:
                logger.error(f"Failed to create InlineImage for {field_name}: {e}")
                # Keep original path as fallback
                continue

    return rendered_context


@task
def create_mep_grouper_page(
    template_path: str,
    output_dir: str,
    context: Dict[str, Any],
    filename: Optional[str] = None,
    validate_images: bool = True,
) -> str:
    """
    Create a Word document from template and context.

    Args:
        template_path: Path to the .docx template file
        output_dir: Directory to save the output document
        context: Dictionary containing template variables and image paths
        filename: Optional output filename (auto-generated if None)
        validate_images: Whether to validate image paths before rendering

    Returns:
        Path to the generated document
    """
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    # Validate paths
    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_directory is empty after normalization")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if not filename:
        subject_name = context.get("name", "unknown")
        # Sanitize subject name for filename
        safe_name = "".join(c for c in subject_name if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_name = safe_name.replace(" ", "_")
        filename = f"{safe_name}_report_{uuid.uuid4().hex[:8]}.docx"

    output_path = Path(output_dir) / filename

    # Validate image paths if requested
    if validate_images:
        image_fields = [
            "profile_photo",
            "mov_map",
            "overview_map",
            "range_map",
            "nsd_plot",
            "speed_plot",
            "collared_event_timeline",
            "mcp_plot",
        ]

        for field_name in image_fields:
            value = context.get(field_name)
            if value and isinstance(value, str):
                path = Path(value)
                if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".html"):
                    if not path.exists():
                        logger.warning(f"Image file not found for {field_name}: {value}")

    # Load template
    try:
        tpl = DocxTemplate(template_path)
        logger.info(f"Loaded template: {template_path}")
    except Exception as e:
        raise ValueError(f"Failed to load templxate: {e}")

    # Prepare context with inline images
    try:
        rendered_context = prepare_mep_context_for_template(
            context=context,
            template=tpl,
        )
        logger.info(f"Prepared context with {len(rendered_context)} fields")
    except Exception as e:
        raise ValueError(f"Failed to prepare context: {e}")

    # Render and save document
    try:
        tpl.render(rendered_context)
        tpl.save(output_path)
        logger.info(f"Saved document to: {output_path}")
        return str(output_path)
    except Exception as e:
        raise ValueError(f"Failed to render or save document: {e}")

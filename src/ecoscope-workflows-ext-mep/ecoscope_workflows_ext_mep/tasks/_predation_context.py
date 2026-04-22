import os
import re
import uuid
import warnings
import pandas as pd
from pathlib import Path
from pydantic import Field
from datetime import datetime
from docxtpl import InlineImage
from docx.shared import Inches
from docxtpl import DocxTemplate
from ecoscope_workflows_core.decorators import task
from typing import Annotated, Optional, Dict, Any
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.skip import SKIP_SENTINEL, SkipSentinel
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

warnings.filterwarnings("ignore")


@task
def clean_string(s: str) -> str:
    s = re.sub(r"[ /\-]", "_", s.lower())
    return re.sub(r"_+", "_", s)


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


@task
def create_mep_subject_context(
    livestock_killed_over_time_by_species: str | SkipSentinel | None,
    livestock_killed_over_time_by_ranch: str | SkipSentinel | None,
    livestock_killed_by_ranch_pie: str | SkipSentinel | None,
    species_by_ranch_heatmap: str | SkipSentinel | None,
    livestock_killed_pie_chart: str | SkipSentinel | None,
    livestock_predation_map: str | SkipSentinel | None,
    density_grid_map: str | SkipSentinel | None,
    species_by_time_heatmap: str | SkipSentinel | None,
    livestock_species_killed_multibar: str | SkipSentinel | None,
    livestock_species_killed_ranch_multibar: str | SkipSentinel | None,
    herder_effectiveness: str | SkipSentinel | None,
    location_of_attack: str | SkipSentinel | None,
    species_ranch_matrix: str | SkipSentinel | None,
    total_livestock_killed_by_ranch: str | SkipSentinel | None,
) -> Dict[str, Any]:
    def unwrap_skip(value):
        """
        Unwrap SkipSentinel values, converting them to None.
        """
        if value is None or value is SKIP_SENTINEL:
            return None
        return value

    def safe_read_csv(file_path: str | None) -> pd.DataFrame:
        """
        Safely read CSV file and return DataFrame.

        Args:
            file_path: Path to CSV file or None

        Returns:
            DataFrame with data, or empty DataFrame if file is invalid
        """
        if file_path is None:
            print("CSV file path is None")
            return pd.DataFrame()

        if not file_path.strip():
            print("CSV file path is empty string")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"CSV file is empty: {file_path}")
            return df
        except FileNotFoundError:
            print(f"CSV file not found: {file_path}")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"CSV file is empty or corrupted: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
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
            print(f"{path_name} is None")
            return None

        if not path.strip():
            print(f"{path_name} is empty string")
            return None

        from pathlib import Path

        if not Path(path).exists():
            print(f"{path_name} does not exist: {path}")
            return None

        return path

    ctx = {}
    return ctx


def is_valid_image(path: str) -> bool:
    """Check image file has a valid PNG or JPEG header (magic bytes)."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return True
        if header[:3] == b"\xff\xd8\xff":
            return True
        return False
    except Exception:
        return False


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


@task
def create_mep_grouper_page(
    template_path: str,
    output_dir: str,
    context: Dict[str, Any],
    filename: Optional[str] = None,
    validate_images: bool = True,
    missing_threshold: int = 7,
) -> Optional[str]:
    """
    Create a Word document from template and context.

    If the subject is missing a number of media metrics greater than or equal to
    `missing_threshold`, the subject is considered to not have sufficient data
    for the reporting period and no document is generated (returns None).

    The following 7 media fields are evaluated for presence:
        - mov_map (speedmap)
        - overview_map (homerange)
        - range_map (seasonal homerange)
        - nsd_plot
        - speed_plot
        - collar_event_timeline
        - mcp_plot

    Args:
        template_path: Path to the .docx template file
        output_dir: Directory to save the output document
        context: Dictionary containing template variables and image paths
        filename: Optional output filename (auto-generated if None)
        validate_images: Whether to validate image paths before rendering
        missing_threshold: Minimum number of missing media items that triggers
            skipping the subject (default: 7, i.e. skip only when *all* media
            items are missing).

    Returns:
        Path to the generated document, or None if the subject was skipped
        due to insufficient metrics.
    """
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_directory is empty after normalization")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    media_fields = [
        "mov_map",
        "overview_map",
        "range_map",
        "nsd_plot",
        "speed_plot",
        "collar_event_timeline",
        "mcp_plot",
    ]

    missing_fields = []
    for field_name in media_fields:
        value = context.get(field_name)
        # Treat as missing if: None, empty/whitespace string, or path doesn't exist on disk
        if value is None:
            missing_fields.append(field_name)
            continue
        if isinstance(value, str):
            if not value.strip():
                missing_fields.append(field_name)
                continue
            if not Path(value).exists():
                missing_fields.append(field_name)
                continue

    subject_name = context.get("name", "unknown")
    total_media = len(media_fields)
    n_missing = len(missing_fields)

    print(
        f"Subject '{subject_name}' media availability: "
        f"{total_media - n_missing}/{total_media} present, "
        f"{n_missing} missing ({missing_fields})"
    )

    if n_missing >= missing_threshold:
        print(
            f"Skipping subject '{subject_name}': all {total_media} media items "
            f"are missing. Subject did not generate metrics for this period."
        )
        return None
    if not filename:
        safe_name = "".join(c for c in subject_name if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_name = safe_name.replace(" ", "_")
        filename = f"{safe_name}_report_{uuid.uuid4().hex[:8]}.docx"

    output_path = Path(output_dir) / filename

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
                        print(f"Image file not found for {field_name}: {value}")

    try:
        tpl = DocxTemplate(template_path)
        print(f"Loaded template: {template_path}")
    except Exception as e:
        raise ValueError(f"Failed to load template: {e}")

    try:
        rendered_context = prepare_mep_context_for_template(
            context=context,
            template=tpl,
        )
        print(f"Prepared context with {len(rendered_context)} fields")
    except Exception as e:
        raise ValueError(f"Failed to prepare context: {e}")

    try:
        tpl.render(rendered_context)
        tpl.save(output_path)
        print(f"Saved document to: {output_path}")
        return str(output_path)
    except Exception as e:
        raise ValueError(f"Failed to render or save document: {e}")

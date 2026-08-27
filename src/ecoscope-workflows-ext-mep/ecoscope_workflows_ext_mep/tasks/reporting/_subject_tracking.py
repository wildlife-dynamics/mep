import pandas as pd
from pathlib import Path
from docx.shared import Inches
from wt_registry import register
from typing import Any, Dict, Optional
from docxtpl import DocxTemplate, InlineImage
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
from ecoscope_workflows_ext_ste.tasks.transformation._tabular import safe_string

def resolve_subject_file_paths(directory: str, subject_name: str) -> Dict[str, Optional[str]]:
    """
    Resolve on-disk paths for a subject's report assets using the
    `{subject_name}_{suffix}` naming convention (e.g. "cherop_speedmap.png").
    Missing files resolve to None.
    """
    base = Path(remove_file_scheme(directory))
    suffix_map = {
        "profile_photo_path": f"{subject_name}.png",
        "subject_info_path": f"{subject_name}_subject_info.csv",
        "speedmap_path": f"{subject_name}_speedmap.png",
        "homerange_map_path": f"{subject_name}_homerange.png",
        "seasonal_homerange_map_path": f"{subject_name}_seasonal_home_range.png",
        "nsd_plot_path": f"{subject_name}_nsd_seasonal_plot.png",
        "speed_plot_path": f"{subject_name}_speed_seasonal_plot.png",
        "collared_event_plot_path": f"{subject_name}_collared_subject_plot.png",
        "mcp_plot_path": f"{subject_name}_mcp_asymptote_plot.png",
        "subject_stats_table_path": f"{subject_name}_subject_stats.csv",
        "subject_occupancy_table_path": f"{subject_name}_subject_occupancy.csv",
    }
    return {
        key: (str(base / fname) if (base / fname).exists() else None)
        for key, fname in suffix_map.items()
    }

def _safe_get(df: Optional[pd.DataFrame], column: str, default: Any = "Undefined") -> Any:
    """Return the first value of a column, or `default` if the frame/column/value is empty/missing."""
    if df is None or df.empty or column not in df.columns:
        return default
    value = df[column].iloc[0]
    if pd.isna(value):
        return default
    return value

def _read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Read a CSV if the path exists, otherwise return None."""
    if path and Path(path).exists():
        return pd.read_csv(path)
    return None

def is_valid_image(path: str) -> bool:
    """Check the file has a valid PNG or JPEG header (magic bytes)."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        return header[:8] == b"\x89PNG\r\n\x1a\n" or header[:3] == b"\xff\xd8\xff"
    except Exception:
        return False

def create_inline_image_inch(
    template: DocxTemplate,
    image_path: str,
    width_cm: float,
    height_cm: float,
) -> InlineImage:
    """Create an InlineImage with the given dimensions."""
    return InlineImage(
        template,
        image_path,
        width=Inches(width_cm),
        height=Inches(height_cm),
    )

def build_subject_context(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """
    Build the full template context for a subject from its report assets.

    Args:
        df: DataFrame containing at least a `subject_name` column (used to
            locate the subject's assets on disk).
        output_dir: Directory containing the subject's generated report assets.

    Returns:
        Context dict with media paths, stats, info and occupancy values.
    """
    subject_name = df["subject_name"].iloc[0]
    paths = resolve_subject_file_paths(directory=output_dir, subject_name=subject_name)

    subject_info = _read_csv(paths["subject_info_path"])
    subject_stats = _read_csv(paths["subject_stats_table_path"])
    subject_occupancy = _read_csv(paths["subject_occupancy_table_path"])

    return {
        # Media paths
        "profile_photo": paths["profile_photo_path"],
        "mov_map": paths["speedmap_path"],
        "overview_map": paths["homerange_map_path"],
        "range_map": paths["seasonal_homerange_map_path"],
        "nsd_plot": paths["nsd_plot_path"],
        "speed_plot": paths["speed_plot_path"],
        "collar_event_timeline": paths["collared_event_plot_path"],
        "mcp_plot": paths["mcp_plot_path"],
        # Subject information
        "name": _safe_get(subject_info, "subject_name"),
        "dob": _safe_get(subject_info, "dob"),
        "sex": _safe_get(subject_info, "sex"),
        "country": _safe_get(subject_info, "country"),
        "status": _safe_get(subject_info, "status_raw"),
        "bio": _safe_get(subject_info, "bio"),
        "distribution": _safe_get(subject_info, "distribution"),
        "id_notes": _safe_get(subject_info, "notes"),
        # Subject statistics
        "mcp": _safe_get(subject_stats, "mcp", 0),
        "etd": _safe_get(subject_stats, "etd", 0),
        "time_tracked_days": _safe_get(subject_stats, "time_tracked_days", 0),
        "time_tracked_years": _safe_get(subject_stats, "time_tracked_years", 0),
        "distance_travelled": _safe_get(subject_stats, "distance_travelled", 0),
        "max_displacement": _safe_get(subject_stats, "max_displacement", 0),
        "night_day_ratio": _safe_get(subject_stats, "night_day_ratio", 0),
        # Occupancy data
        "national_pa_use": _safe_get(subject_occupancy, "national_pa_use", 0),
        "community_pa_use": _safe_get(subject_occupancy, "community_pa_use", 0),
        "crop_raid_percent": _safe_get(subject_occupancy, "crop_raid_percent", 0),
        "kenya_use": _safe_get(subject_occupancy, "kenya_use", 0),
        "unprotected": _safe_get(subject_occupancy, "unprotected", 0),
    }

IMAGE_FIELD_MAPPING = {
    "collar_event_timeline": {"height": 1.58, "width": 10.54},
    "nsd_plot": {"height": 2.61, "width": 10.58},
    "speed_plot": {"height": 2.61, "width": 10.58},
    "mcp_plot": {"height": 2.61, "width": 10.58},
    "range_map": {"height": 7.08, "width": 5.36},
    "mov_map": {"height": 4.42, "width": 7.47},
    "overview_map": {"height": 7.06, "width": 5.4},
    "profile_photo": {"height": 3.69, "width": 3.54},
}

def prepare_context(context: Dict[str, Any], template: DocxTemplate) -> Dict[str, Any]:
    """Convert image paths in the context into InlineImage objects."""
    rendered_context = context.copy()
    for field_name, dims in IMAGE_FIELD_MAPPING.items():
        if field_name not in rendered_context:
            continue
        image_path = rendered_context[field_name]
        if not image_path:
            print(f"Empty image path for field: {field_name}")
            continue
        if not Path(image_path).exists():
            print(f"Image file not found for {field_name}: {image_path}")
            rendered_context[field_name] = None
            continue
        if not is_valid_image(image_path):
            print(f"Invalid image format for {field_name}: {image_path}")
            rendered_context[field_name] = None
            continue
        try:
            rendered_context[field_name] = create_inline_image_inch(
                template=template,
                image_path=image_path,
                width_cm=dims["width"],
                height_cm=dims["height"],
            )
            print(f"Created InlineImage for {field_name}: {dims['width']}x{dims['height']} cm")
        except Exception as e:
            print(f"Failed to create InlineImage for {field_name}: {e}")
    return rendered_context

@register()
def generate_subject_report(
    df: AnyDataFrame,
    output_dir: str,
    template_path: str,
) -> str:
    """
    Generate a rendered subject report .docx.

    Args:
        df: DataFrame with a `subject_name` column identifying the subject.
        output_dir: Directory containing the subject's generated assets.
        template_path: Path to the .docx template.

    Returns:
        The path to the saved document.
    """
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)
    
    subject_name = df["subject_name"].iloc[0]
    subject_name = safe_string(value=subject_name)
    output_path = str(Path(output_dir) / f"{subject_name}.docx")
    context = build_subject_context(df=df, output_dir=output_dir)
    template = DocxTemplate(template_path)
    rendered_context = prepare_context(context=context, template=template)
    template.render(rendered_context)
    template.save(output_path)
    print(f"Saved report to {output_path}")
    return output_path
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from docx.shared import Inches
from typing import Optional,Union,Dict,Any
from docxtpl import DocxTemplate, InlineImage
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

def get_image_dimensions_from_pixels(
    image_path: str,
    dpi: int = 96,
    max_dimension_inches: float = 1.5,
) -> tuple[float, float]:
    """
    Calculate image dimensions in inches based on pixel dimensions,
    scaled so the largest dimension fits within max_dimension_inches.
    Preserves aspect ratio for both wide and square images.
    """
    with Image.open(image_path) as img:
        width_pixels, height_pixels = img.size

        # Get DPI from image metadata
        image_dpi = img.info.get("dpi", (dpi, dpi))
        if isinstance(image_dpi, tuple):
            dpi_x, dpi_y = image_dpi
        else:
            dpi_x = dpi_y = dpi

    width_inches = width_pixels / dpi_x
    height_inches = height_pixels / dpi_y

    if width_inches > height_inches:
        scale_factor = max_dimension_inches / width_inches
    else:
        scale_factor = max_dimension_inches / height_inches

    return width_inches * scale_factor, height_inches * scale_factor

@task
def generate_source_voltage_report(
    org_logo_path: Union[str, Path, None],
    report_period: TimeRange,
    prepared_by: str,
    output_dir: Union[str, Path],
    template_path: Union[str, Path],
    filename: Optional[str] = None,
) -> str:
    """
    Generate a source voltage Word report from a template.

    Args:
        org_logo_path:  Path to the organisation logo (file:// URIs are accepted).
        report_period:  Time range covered by the report.
        prepared_by:    Name of the person who prepared the report.
        output_dir:     Directory that contains the chart images and receives the output file.
        template_path:  Path to the .docx template.
        filename:       Optional output filename; auto-generated when omitted.

    Returns:
        Absolute path of the saved report.
    """
    print("=" * 80)
    print("Generating Source Voltage Report …")

    # ── Resolve & validate paths ───────────────────────────────────────────
    template_path = remove_file_scheme(str(template_path))
    output_dir    = remove_file_scheme(str(output_dir))

    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Template Path  : {template_path}")
    print(f"Output Directory: {output_dir}")

    # ── Resolve logo ───────────────────────────────────────────────────────
    resolved_logo_path: Optional[str] = None
    if org_logo_path is not None:
        resolved_logo_path = remove_file_scheme(str(org_logo_path))
        if not resolved_logo_path.strip():
            raise ValueError("org_logo_path is empty after normalization")

    # ── Discover chart images ──────────────────────────────────────────────
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    images_found: Dict[str, str] = {}

    for root, _, files in os.walk(output_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTS:
                parts = p.stem.split("_")
                key = parts[1] if len(parts) > 1 else p.stem
                images_found[key] = str(p)

    print(f"Found {len(images_found)} image file(s)")

    # ── Build template context ─────────────────────────────────────────────
    tpl = DocxTemplate(template_path)
    context: Dict[str, Any] = {}

    # Per-collar voltage charts
    context["source_voltage_charts"] = [
        {
            "source_voltage_image": InlineImage(
                tpl, path, width=Inches(6.58), height=Inches(3.85)
            ),
            "subject": subject_name,
        }
        for subject_name, path in images_found.items()
    ]

    # Organisation logo
    if resolved_logo_path:
        logo_width, logo_height = get_image_dimensions_from_pixels(
            resolved_logo_path,
            dpi=125,
            max_dimension_inches=1.5,
        )
        print(f"Logo dimensions: {logo_width:.2f}\" × {logo_height:.2f}\"")
        context["org_logo"] = InlineImage(
            tpl,
            resolved_logo_path,
            width=Inches(logo_width),
            height=Inches(logo_height),
        )
    else:
        context["org_logo"] = None

    # Metadata
    context["prepared_by"] = prepared_by

    if report_period:
        fmt = getattr(report_period, "time_format", "%Y-%m-%d")
        context["report_period"] = (
            f"{report_period.since.strftime(fmt)} to {report_period.until.strftime(fmt)}"
        )
    else:
        context["report_period"] = None

    context["time_generated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Render & save ──────────────────────────────────────────────────────
    output_filename = (
        filename
        or f"source_voltage_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.docx"
    )
    output_path = os.path.join(output_dir, output_filename)

    tpl.render(context)
    tpl.save(output_path)

    print("\nSource voltage report generated successfully!")
    print(f"Output: {output_path}")
    print("=" * 80)

    return str(output_path)
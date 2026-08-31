import os
import pandas as pd
from pathlib import Path
from docx.shared import Inches
from wt_registry import register
from docxtpl import InlineImage
from docxtpl import DocxTemplate
from typing import Optional, Dict, Any, Union
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
VOLTAGE_CHART_SUFFIX = "_historic_voltage"
SINGLE_IMAGE_STEMS = {
    "elephant_speedmap": "elephant_speedmap",
    "elephant_sighting_map": "elephant_sightings_map",
    "vehicle_patrol_tracks": "vehicle_patrols_map",
    "foot_patrol_tracks": "foot_patrols_map",
}
SITREP_CSV_STEM = "sitrep_report"


def _read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Read a CSV if the path exists, otherwise return None."""
    if path and Path(path).exists():
        return pd.read_csv(path)
    return None


def _maybe_image(tpl: DocxTemplate, path: Optional[str]) -> Optional[InlineImage]:
    if not path:
        return None  # template must guard with {% if %}
    return InlineImage(tpl, path, width=Inches(6.58), height=Inches(3.85))


@register()
def create_mep_monthly_report(
    template_path: Union[str, Path],
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> str:
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    if not filename:
        filename = f"mep_monthly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.docx"
    output_path = Path(output_dir) / filename

    single_image_paths: Dict[str, str] = {}
    voltage_images_found: list[tuple[str, str]] = []
    sitrep_csv_path: Optional[str] = None

    for root, _, files in os.walk(output_dir):
        for f in sorted(files):
            p = Path(root) / f
            stem = p.stem
            suffix = p.suffix.lower()

            if suffix == ".csv" and stem == SITREP_CSV_STEM:
                sitrep_csv_path = str(p)
                continue

            if suffix not in IMAGE_EXTS:
                continue

            if stem.endswith(VOLTAGE_CHART_SUFFIX):
                subject_name = stem[: -len(VOLTAGE_CHART_SUFFIX)]
                voltage_images_found.append((subject_name, str(p)))
                continue

            for context_key, expected_stem in SINGLE_IMAGE_STEMS.items():
                if stem == expected_stem:
                    single_image_paths[context_key] = str(p)
                    break

    print(
        f"Found {len(single_image_paths)} single chart(s), "
        f"{len(voltage_images_found)} voltage chart(s), "
        f"sitrep_csv={'found' if sitrep_csv_path else 'missing'}"
    )
    print(f"Single image keys: {single_image_paths.keys()}")

    try:
        tpl = DocxTemplate(template_path)
        print(f"Loaded template: {template_path}")
    except Exception as e:
        raise ValueError(f"Failed to load template: {e}")

    collar_voltage_list = [
        {
            "collar_voltage_image": InlineImage(tpl, path, width=Inches(6.58), height=Inches(3.85)),
            "subject": subject_name,
        }
        for subject_name, path in voltage_images_found
    ]

    sitrep_df = _read_csv(sitrep_csv_path)
    sitrep = sitrep_df.to_dict(orient="records") if sitrep_df is not None else []

    context: Dict[str, Any] = {
        "elephant_speedmap": _maybe_image(tpl, single_image_paths.get("elephant_speedmap")),  # elephant_speedmap
        "elephant_sighting_map": _maybe_image(tpl, single_image_paths.get("elephant_sighting_map")),
        "vehicle_patrol_tracks": _maybe_image(tpl, single_image_paths.get("vehicle_patrol_tracks")),
        "foot_patrol_tracks": _maybe_image(tpl, single_image_paths.get("foot_patrol_tracks")),
        "sitrep": sitrep,
        "collar_voltage_list": collar_voltage_list,
    }

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        tpl.render(context)
        tpl.save(output_path)
        print(f"Saved document to: {output_path}")
        return str(output_path)
    except Exception as e:
        raise ValueError(f"Failed to render or save document: {e}")

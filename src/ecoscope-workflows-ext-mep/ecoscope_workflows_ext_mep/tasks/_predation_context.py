import os
import re
import textwrap
import pandas as pd
from typing import Optional
from docx.shared import Inches
from ecoscope_workflows_core.decorators import task
from docxtpl import DocxTemplate, InlineImage
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme


def categorize_files(files: list[str]) -> dict:
    patterns = {
        "images": {
            "livestock_killed_over_time_by_species": r"_livestock_killed_over_time_by_species_chart\.png",
            "livestock_killed_over_time_by_ranch": r"_livestock_killed_over_time_by_ranch_chart\.png",
            "livestock_killed_by_ranch_pie": r"_livestock_killed_by_ranch_pie_chart\.png",
            "species_by_ranch_heatmap": r"_species_by_ranch_heatmap\.png",
            "livestock_killed_pie_chart": r"_livestock_killed_pie_chart\.png",
            "livestock_predation_map": r"_livestock_predation_map\.png",
            "density_grid_map": r"_density_grid_map\.png",
            "species_by_time_heatmap": r"_species_by_time_heatmap\.png",
        },
        "long_images": {
            "livestock_species_killed_multibar": r"_livestock_species_killed_multibar\.png",
            "livestock_species_killed_ranch_multibar": r"_livestock_species_killed_ranch_multibar\.png",
        },
        "csvs": {
            "herder_effectiveness": r"herder_effectiveness_.*\.csv",
            "location_of_attack": r"location_of_attack_.*\.csv",
            "species_ranch_matrix": r"species_ranch_matrix_.*\.csv",
            "total_livestock_killed_by_ranch": r"total_livestock_killed_by_ranch_.*\.csv",
        },
    }

    result = {group: {} for group in patterns}

    for group, group_patterns in patterns.items():
        for key, pattern in group_patterns.items():
            match = next((f for f in files if re.search(pattern, f)), None)
            if match:
                result[group][key] = match

    return result

def _load_csv_to_records(
    output_dir: str,
    csv_filename: Optional[str],
    context_key: str,
    drop_columns: Optional[list[str]] = None,
) -> Optional[list[dict]]:
    """Load a CSV, clean it, and return records. Returns None if missing or invalid."""
    if not csv_filename:
        print(f"Warning: No CSV filename provided for '{context_key}'")
        return None
    
    csv_path = os.path.join(output_dir, csv_filename)
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found for '{context_key}': {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Warning: Could not parse CSV for '{context_key}': {type(e).__name__}: {e}")
        return None
    
    if df.empty:
        print(f"Warning: CSV is empty for '{context_key}'")
        return None
    
    # Drop unwanted columns if they exist
    if drop_columns:
        cols_to_drop = [c for c in drop_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
    
    # Convert numeric columns to int (handle NaN safely)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0).astype(int)
    
    return df.to_dict(orient="records")

@task
def generate_predation_report(
    template_path: str,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)
    
    print(f"\nTemplate Path: {template_path}")
    print(f"Output Directory: {output_dir}")
    
    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_directory is empty after normalization")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(output_dir)
    output = categorize_files(files)
    
    # Build lookup: stem (no extension) -> full path
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    images_found = {
        os.path.splitext(f)[0]: os.path.join(output_dir, f)
        for f in files
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    }
    
    tpl = DocxTemplate(template_path)
    context = {}
    
    # --- Images ---
    image_specs = [
        (output.get("images", {}), Inches(3.12)),
        (output.get("long_images", {}), Inches(8.1)),
    ]
    for image_dict, height in image_specs:
        for template_var, file_stem in image_dict.items():
            key = os.path.splitext(file_stem)[0]
            if key not in images_found:
                print(f"Warning: Image not found for '{template_var}' (key: {key})")
                context[template_var] = None
                continue
            try:
                context[template_var] = InlineImage(
                    tpl,
                    images_found[key],
                    width=Inches(5.34),
                    height=height,
                )
            except Exception as e:
                print(f"Warning: Could not load image '{template_var}': {type(e).__name__}: {e}")
                context[template_var] = None
    
    # --- CSVs ---
    csvs = output.get("csvs", {})
    csv_keys = [
        "species_ranch_matrix",
        "total_livestock_killed_by_ranch",
        "location_of_attack",
        "herder_effectiveness",
    ]
    for key in csv_keys:
        context[key] = _load_csv_to_records(
            output_dir,
            csvs.get(key),
            context_key=key,
            drop_columns=["Unnamed: 0"],
        )
    
    # --- Render ---
    output_filename = filename or f"predation_sect_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.docx"
    output_path = os.path.join(output_dir, output_filename)
    tpl.render(context)
    tpl.save(output_path)
    return output_path
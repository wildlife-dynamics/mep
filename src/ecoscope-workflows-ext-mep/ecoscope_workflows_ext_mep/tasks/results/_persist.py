import os 
import ecoscope 
import hashlib
import pandas as pd 
from pathlib import Path 
from wt_registry import register
from typing import Optional,Union,Annotated,Field,Dict
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

def safe_string(
    value: Annotated[str, Field(description="String to make safe for use as a filename")],
) -> str:
    """Sanitize a string for filenames: replace spaces with underscores, remove special characters, lowercase."""
    import re

    safe = re.sub(r"[^\w\s-]", "", value)
    safe = re.sub(r"\s+", "_", safe)
    return safe.lower().strip("_")

@register()
def persist_subject_photo(
    subject_df: AnyDataFrame,
    column: str,
    filename_column: Optional[str] = None,
    output_path: Union[str, Path] = None,
    image_type: str = ".png",
    overwrite_existing: bool = True,
) -> list[str]:
    """Download photos from `column`, naming files after `filename_column`
    when provided, else a URL-hash-based default. Returns list of saved paths."""

    def extract_url(value) -> Optional[str]:
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, dict):
            for key in ("url", "href", "link", "value", "src"):
                if key in value:
                    candidate = str(value[key]).strip()
                    if candidate.startswith(("http://", "https://")):
                        return candidate
            for v in value.values():
                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    return v.strip()
        return None

    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    output_path = Path(remove_file_scheme(str(output_path).strip()))
    output_path.mkdir(parents=True, exist_ok=True)

    if column not in subject_df.columns:
        print(f"Column '{column}' not found. Available: {list(subject_df.columns)}")
        return []

    if filename_column is not None and filename_column not in subject_df.columns:
        print(
            f"Filename column '{filename_column}' not found; "
            f"falling back to hash-based names. Available: {list(subject_df.columns)}"
        )
        filename_column = None

    if not image_type.startswith("."):
        image_type = f".{image_type}"

    persisted_paths: list[str] = []
    used_names: Dict[str, int] = {}

    for idx, row in subject_df.iterrows():
        raw = row[column]
        if not isinstance(raw, (dict, list)) and pd.isna(raw):
            continue

        url = extract_url(raw)
        if not url or not url.startswith(("http://", "https://")):
            print(f"Skipping invalid URL at index {idx}: {raw!r}")
            continue

        filename = None
        if filename_column is not None:
            name_value = row[filename_column]
            if isinstance(name_value, str) and name_value.strip():
                filename = safe_string(name_value)

        if not filename:
            filename = f"profile_photo_{hashlib.sha256(url.encode()).hexdigest()[:8]}"

        count = used_names.get(filename, 0)
        used_names[filename] = count + 1
        if count > 0:
            filename = f"{filename}_{count}"

        file_path = output_path / f"{filename}{image_type}"

        try:
            processed_url = url.replace("dl=0", "dl=1") if "dropbox.com" in url else url
            ecoscope.io.utils.download_file(processed_url, str(file_path), overwrite_existing)
            print(f"Downloaded photo for '{filename}' to {file_path}")
            persisted_paths.append(str(file_path))
        except Exception as e:
            print(f"Error processing URL at index {idx} ({url}): {e}")
            continue

    return persisted_paths
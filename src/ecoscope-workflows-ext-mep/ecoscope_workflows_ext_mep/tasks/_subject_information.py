import os
import hashlib
import ecoscope
import logging
import pandas as pd
from pydantic import Field
from pathlib import Path
from datetime import datetime
from ecoscope_workflows_core.decorators import task
from typing import Union, Annotated, cast, Optional, Dict
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerClient
from ecoscope_workflows_core.annotations import AdvancedField, AnyDataFrame
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

logger = logging.getLogger(__name__)


@task
def get_subject_df(
    client: EarthRangerClient,
    include_inactive: Annotated[
        bool,
        AdvancedField(default=None, description="Include inactive subjects in the list."),
    ] = None,
    bbox: Annotated[
        tuple[float, float, float, float] | None,
        Field(
            description="Bounding box filter as (west, south, east, north). "
            "Includes subjects with track data inside the box."
        ),
    ] = None,
    subject_group_id: Annotated[str | None, Field(description="Subject group ID to filter subjects by.")] = None,
    subject_group_name: Annotated[str | None, Field(description="Subject group name to filter subjects by.")] = None,
    name: Annotated[str | None, Field(description="Filter subjects by name.")] = None,
    updated_since: Annotated[
        str | None, Field(description="Only include subjects updated since this timestamp (ISO).")
    ] = None,
    updated_until: Annotated[
        str | None, Field(description="Only include subjects updated until this timestamp (ISO).")
    ] = None,
    tracks: Annotated[bool | None, Field(description="Whether to include recent tracks for each subject.")] = None,
    ids: Annotated[
        list[str] | None,
        Field(description="List of subject IDs to fetch. Splits requests in chunks if large."),
    ] = None,
    max_ids_per_request: Annotated[
        int,
        Field(description="Maximum number of IDs per request when splitting batched subject queries."),
    ] = 50,
    raise_on_empty: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Whether to abort the workflow if no subjects are returned from EarthRanger.",
        ),
    ] = True,
) -> AnyDataFrame:
    """Fetch subjects from EarthRanger with filtering options."""

    df = client.get_subjects(
        include_inactive=include_inactive,
        bbox=bbox,
        subject_group_id=subject_group_id,
        subject_group_name=subject_group_name,
        name=name,
        updated_since=updated_since,
        updated_until=updated_until,
        tracks=tracks,
        id=",".join(ids) if ids else None,
        max_ids_per_request=max_ids_per_request,
    )

    if raise_on_empty and df.empty:
        raise ValueError("No data returned from EarthRanger for get_subjects")
    return df


@task
def persist_subject_photo(
    subject_df: AnyDataFrame,
    column: str,
    output_path: Union[str, Path] = None,
    image_type: str = ".png",
    overwrite_existing: bool = True,
) -> Optional[str]:
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = remove_file_scheme(output_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if column not in subject_df.columns:
        print(f"Column '{column}' not found. Available: {list(subject_df.columns)}")
        return None 
    
    last_file_path: Optional[Path] = None
    for idx, url in subject_df[column].dropna().items():
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            logger.warning(f"Skipping invalid URL at index {idx}: {url}")
            print(f"Skipping invalid URL at index {idx}: {url}")
            continue

        try:
            df_subset = subject_df.loc[[idx]]
            row_hash = hashlib.sha256(pd.util.hash_pandas_object(df_subset, index=True).values).hexdigest()
            filename = f"profile_photo_{row_hash[:3]}"

            if not image_type.startswith("."):
                image_type = f".{image_type}"

            file_path = output_path / f"{filename}{image_type}"
            processed_url = url.replace("dl=0", "dl=1") if "dropbox.com" in url else url
            ecoscope.io.utils.download_file(processed_url, str(file_path), overwrite_existing)
            logger.info(f"Downloaded profile photo for index {idx} to {file_path}")
            print(f"Downloaded profile photo for index {idx} to {file_path}")
            last_file_path = file_path
        except Exception as e:
            print(f"Error processing URL at index {idx} ({url}): {e}")
            continue

    return str(last_file_path) if last_file_path else None


def safe_strip(x) -> str:
    return "" if x is None else str(x).strip()


def truncate_at_sentence(text: str, maxlen: int) -> str:
    if len(text) <= maxlen:
        return text
    cut = text[:maxlen]
    dot = cut.rfind(".")
    return (cut[: dot + 1] if dot >= 40 else cut.rstrip()) + ("..." if dot < 40 else "")


def format_date(date_str: str) -> str:
    s = safe_strip(date_str)
    if not s:
        return ""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%b %d, %Y", "%d %b %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%d %b %Y")
        except ValueError:
            continue
    return s


@task
def process_subject_information(
    subject_df: AnyDataFrame, output_path: Union[str, Path], maxlen: int = 1000
) -> Optional[AnyDataFrame]:
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()
    output_path = remove_file_scheme(output_path)
    os.makedirs(output_path, exist_ok=True)

    if subject_df.empty:
        empty_cols = ["subject_name", "dob", "sex", "country", "notes", "status", "status_raw", "bio", "distribution"]
        return cast(AnyDataFrame, pd.DataFrame([{col: "" for col in empty_cols}]))

    def process_single_subject(row: pd.Series) -> Dict[str, str]:
        bio = safe_strip(row.get("subject_bio", ""))
        if len(bio) > maxlen:
            bio = truncate_at_sentence(bio, maxlen)

        status_value = safe_strip(row.get("status", ""))
        status_color = "green" if status_value.lower() == "active" else "red"
        dob_formatted = format_date(safe_strip(row.get("date_of_birth", "")))

        subject_info = {
            "subject_name": safe_strip(row.get("subject_name", "")).title(),
            "dob": dob_formatted,
            "sex": safe_strip(row.get("subject_sex", "")).capitalize(),
            "country": safe_strip(row.get("country", "")),
            "notes": safe_strip(row.get("notes", "")),
            "status": f'<span style="color: {status_color};">{status_value}</span>',
            "status_raw": status_value,
            "bio": bio,
            "distribution": safe_strip(row.get("distribution", "")),
        }
        return {k: ("" if v is None else str(v)) for k, v in subject_info.items()}

    processed_records = [process_single_subject(row) for _, row in subject_df.iterrows()]
    return cast(AnyDataFrame, pd.DataFrame(processed_records))

import math
import html
import pandas as pd
from typing import cast, Dict
from datetime import datetime
from wt_registry import register
from ecoscope.platform.annotations import AnyDataFrame


def safe_strip(x) -> str:
    """Return '' for None/NaN, else the stripped string."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none", "nat") else s


def scalar(row: pd.Series, key: str, default=""):
    """Get a scalar from a row even when duplicate column names make row[key] a Series."""
    value = row.get(key, default)
    if isinstance(value, pd.Series):
        non_null = value.dropna()
        value = non_null.iloc[0] if not non_null.empty else default
    return value


def truncate_at_sentence(text: str, maxlen: int, min_sentence_pos: int = 40) -> str:
    if len(text) <= maxlen:
        return text
    cut = text[:maxlen]
    dot = cut.rfind(".")
    if dot >= min_sentence_pos:
        return cut[: dot + 1]
    return cut.rstrip() + "..."


def format_date(date_str) -> str:
    s = safe_strip(date_str)
    if not s:
        return ""
    # Year-only values, incl. floats like 1985.0 coming from pandas
    try:
        f = float(s)
        if f.is_integer() and 1000 <= f <= 9999:
            return str(int(f))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%b %d, %Y", "%d %b %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%d %b %Y")
        except ValueError:
            continue
    return s


OUTPUT_COLUMNS = [
    "subject_name",
    "dob",
    "sex",
    "country",
    "notes",
    "status",
    "status_raw",
    "bio",
    "distribution",
]

STATUS_COLORS = {"active": "green"}  # everything else -> red


@register()
def process_subject_information(
    subject_df: AnyDataFrame,
    maxlen: int = 1000,
) -> AnyDataFrame:
    """Normalize subject fields for display (formatted dates, truncated bio, status badge)."""

    if subject_df.empty:
        return cast(AnyDataFrame, pd.DataFrame(columns=OUTPUT_COLUMNS))

    def process_single_subject(row: pd.Series) -> Dict[str, str]:
        bio = truncate_at_sentence(safe_strip(scalar(row, "subject_bio")), maxlen)

        status_value = safe_strip(scalar(row, "status"))
        status_color = STATUS_COLORS.get(status_value.lower(), "red")
        status_html = f'<span style="color: {status_color};">{html.escape(status_value)}</span>' if status_value else ""

        return {
            "subject_name": safe_strip(scalar(row, "subject_name")).title(),
            "dob": format_date(scalar(row, "date_of_birth")),
            "sex": safe_strip(scalar(row, "subject_sex")).capitalize(),
            "country": safe_strip(scalar(row, "country")),
            "notes": safe_strip(scalar(row, "notes")),
            "status": status_html,
            "status_raw": status_value,
            "bio": bio,
            "distribution": safe_strip(scalar(row, "distribution")),
        }

    processed_records = [process_single_subject(row) for _, row in subject_df.iterrows()]
    return cast(AnyDataFrame, pd.DataFrame(processed_records, columns=OUTPUT_COLUMNS))

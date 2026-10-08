from itertools import combinations
from typing import Annotated

import pandas as pd
from ecoscope.platform.annotations import AnyDataFrame
from pydantic import Field
from wt_registry import register

from ..results._subject_source import (
    _first,
    _source_label,
    assignment_bounds,
    format_assignment_range,
)

ISSUE_COLUMNS = ["Issue", "Record type", "Name", "ID", "Detail"]

NO_SOURCE = "Subject has no source"
NO_SUBJECT = "Source has no subject"
BAD_DATE = "Assignment has an out-of-range date"
MISSING_SOURCE = "Assignment points to a missing source"
OVERLAP = "Source assigned to two subjects at once"

# Order rows appear in, most actionable first.
ISSUE_ORDER = [BAD_DATE, OVERLAP, MISSING_SOURCE, NO_SOURCE, NO_SUBJECT]


def _ids(df: pd.DataFrame, column: str = "id") -> pd.Series:
    return df[column].astype(str) if column in df else pd.Series(dtype=str)


def _detail(row: pd.Series, columns: tuple[str, ...]) -> str:
    return ", ".join(str(v) for col in columns if (v := _first(row, col)) is not None)


def _overlaps(a: dict, b: dict) -> bool:
    # Open ends are unbounded. Strict comparison so a hand-over on the same instant
    # (one assignment ends as the next starts) is not an overlap.
    a_start, a_end = (
        a["start"] or pd.Timestamp.min.tz_localize("UTC"),
        a["end"] or pd.Timestamp.max.tz_localize("UTC"),
    )
    b_start, b_end = (
        b["start"] or pd.Timestamp.min.tz_localize("UTC"),
        b["end"] or pd.Timestamp.max.tz_localize("UTC"),
    )
    return a_start < b_end and b_start < a_end


@register()
def get_subject_source_issues(
    subjects: AnyDataFrame,
    subjectsources: AnyDataFrame,
    sources: AnyDataFrame,
    check_overlaps: Annotated[
        bool,
        Field(
            description="Flag sources assigned to more than one subject over the same period."
        ),
    ] = True,
) -> AnyDataFrame:
    """List subject-source configuration problems, one row per issue.

    Checks, in the order rows are returned:

    - **Assignment has an out-of-range date**: a start or end date that can't be read
      (e.g. a year typo like 0023). ER's open-start (0001) and open-end (9999)
      placeholders are not flagged.
    - **Source assigned to two subjects at once**: two assignments of the same source
      to different subjects whose date ranges overlap (if `check_overlaps`).
    - **Assignment points to a missing source**: the assignment's source is not in
      `sources`.
    - **Subject has no source**: a subject with no assignment.
    - **Source has no subject**: a source with no assignment to any of `subjects`.

    Only assignments of subjects in `subjects` are considered, matching the diagram:
    when inactive subjects are excluded upstream, their assignments are ignored and a
    source assigned only to them is reported as having no subject (the detail says so).

    Returns:
        A dataframe with columns Issue, Record type, Name, ID and Detail; empty (with
        those columns) when nothing is wrong.
    """
    subject_ids = set(_ids(subjects))
    source_ids = set(_ids(sources))
    subject_names = {
        str(r["id"]): str(_first(r, "name", "id")) for _, r in subjects.iterrows()
    }
    source_names = {str(r["id"]): _source_label(r) for _, r in sources.iterrows()}

    links = subjectsources.copy()
    links["subject"] = _ids(links, "subject")
    links["source"] = _ids(links, "source")
    all_assigned_sources = set(links["source"])
    links = links[links["subject"].isin(subject_ids)]

    rows: list[dict] = []
    assignments: list[dict] = []
    for _, link in links.iterrows():
        start, end, bad_start, bad_end = assignment_bounds(link)
        subject_name = subject_names[link["subject"]]
        source_name = source_names.get(link["source"], link["source"])
        name = f"{subject_name} → {source_name}"
        link_id = str(_first(link, "id") or "")
        if bad_start or bad_end:
            parts = [
                f"start date {bad_start}" if bad_start else "",
                f"end date {bad_end}" if bad_end else "",
            ]
            rows.append(
                {
                    "Issue": BAD_DATE,
                    "Record type": "Assignment",
                    "Name": name,
                    "ID": link_id,
                    "Detail": " and ".join(p for p in parts if p).capitalize()
                    + " is out of range",
                }
            )
        if link["source"] not in source_ids:
            rows.append(
                {
                    "Issue": MISSING_SOURCE,
                    "Record type": "Assignment",
                    "Name": name,
                    "ID": link_id,
                    "Detail": f"Source {link['source']} is not in the sources list",
                }
            )
        if not (bad_start or bad_end):
            assignments.append(
                {
                    "subject": link["subject"],
                    "source": link["source"],
                    "start": start,
                    "end": end,
                    "label": format_assignment_range(start, end, None, None),
                }
            )

    if check_overlaps:
        by_source: dict[str, list[dict]] = {}
        for a in assignments:
            by_source.setdefault(a["source"], []).append(a)
        for source_id, group in by_source.items():
            for a, b in combinations(group, 2):
                if a["subject"] != b["subject"] and _overlaps(a, b):
                    rows.append(
                        {
                            "Issue": OVERLAP,
                            "Record type": "Source",
                            "Name": source_names.get(source_id, source_id),
                            "ID": source_id,
                            "Detail": (
                                f"{subject_names[a['subject']]} ({a['label']}) and "
                                f"{subject_names[b['subject']]} ({b['label']})"
                            ),
                        }
                    )

    linked_subjects = set(links["subject"])
    for _, subject in subjects.iterrows():
        sid = str(subject["id"])
        if sid not in linked_subjects:
            rows.append(
                {
                    "Issue": NO_SOURCE,
                    "Record type": "Subject",
                    "Name": subject_names[sid],
                    "ID": sid,
                    "Detail": _detail(subject, ("subject_type", "subject_subtype")),
                }
            )

    linked_sources = set(links["source"])
    for _, source in sources.iterrows():
        src_id = str(source["id"])
        if src_id not in linked_sources:
            detail = _detail(source, ("provider", "source_type", "model_name"))
            if src_id in all_assigned_sources:
                note = "Only assigned to subjects not in this report (e.g. inactive)"
                detail = f"{detail}. {note}" if detail else note
            rows.append(
                {
                    "Issue": NO_SUBJECT,
                    "Record type": "Source",
                    "Name": source_names[src_id],
                    "ID": src_id,
                    "Detail": detail,
                }
            )

    issues = pd.DataFrame(rows, columns=ISSUE_COLUMNS)
    if issues.empty:
        return issues
    issues["_order"] = issues["Issue"].map(ISSUE_ORDER.index)
    issues = issues.sort_values(
        ["_order", "Name"],
        key=lambda col: col.str.casefold() if col.dtype == object else col,
    )
    return issues.drop(columns="_order").reset_index(drop=True)

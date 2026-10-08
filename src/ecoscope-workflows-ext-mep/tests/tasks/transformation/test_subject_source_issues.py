"""Tests for ecoscope_workflows_ext_mep.tasks.transformation._subject_source_issues.

`get_subject_source_issues` is registered via `wt_registry.register()`, a no-op
decorator at call time, so it is called directly against small hand-built frames
shaped like EarthRanger's subjects/sources/subjectsources responses, mirroring the
scenarios in the subject tracking QA workflow's test-cases.yaml.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_mep.tasks.transformation import get_subject_source_issues
from ecoscope_workflows_ext_mep.tasks.transformation._subject_source_issues import (
    BAD_DATE,
    ISSUE_COLUMNS,
    MISSING_SOURCE,
    NO_SOURCE,
    NO_SUBJECT,
    OVERLAP,
)

OPEN_END = "9999-12-31T23:59:59+00:00"


def _link(id_, subject, source, lower, upper):
    return {
        "id": id_,
        "subject": subject,
        "source": source,
        "assigned_range": {"lower": lower, "upper": upper},
    }


@pytest.fixture
def subjects():
    return pd.DataFrame(
        [
            {
                "id": "amani",
                "name": "Amani",
                "subject_type": "wildlife",
                "subject_subtype": "elephant",
                "is_active": True,
            },
            {
                "id": "baraka",
                "name": "Baraka",
                "subject_type": "wildlife",
                "subject_subtype": "elephant",
                "is_active": True,
            },
            {
                "id": "zawadi",
                "name": "Zawadi",
                "subject_type": "wildlife",
                "subject_subtype": "giraffe",
                "is_active": False,
            },
            {
                "id": "truck",
                "name": "Ranger Truck 3",
                "subject_type": "vehicle",
                "subject_subtype": "car",
                "is_active": True,
            },
        ]
    )


@pytest.fixture
def sources():
    return pd.DataFrame(
        [
            {"id": "st1001", "manufacturer_id": "ST-1001", "provider": "savannah"},
            {"id": "st0950", "manufacturer_id": "ST-0950", "provider": "savannah"},
            {"id": "st1002", "manufacturer_id": "ST-1002", "provider": "savannah"},
            {"id": "awt2001", "manufacturer_id": "AWT-2001", "provider": "awt"},
            {"id": "gsm3001", "manufacturer_id": "GSM-3001", "provider": "gsm"},
        ]
    )


@pytest.fixture
def subjectsources():
    return pd.DataFrame(
        [
            _link("a1", "amani", "st1001", "2024-03-01T00:00:00+00:00", OPEN_END),
            _link(
                "a2",
                "amani",
                "st0950",
                "2022-01-15T00:00:00+00:00",
                "2024-02-28T00:00:00+00:00",
            ),
            _link("b1", "baraka", "st1002", "2023-06-10T00:00:00+00:00", OPEN_END),
            _link(
                "z1",
                "zawadi",
                "awt2001",
                "2021-05-01T00:00:00+00:00",
                "2022-11-30T00:00:00+00:00",
            ),
        ]
    )


def _issues(df, issue):
    return df[df["Issue"] == issue]


class TestBaseCase:
    def test_columns(self, subjects, subjectsources, sources):
        issues = get_subject_source_issues(subjects, subjectsources, sources)
        assert list(issues.columns) == ISSUE_COLUMNS

    def test_flags_subject_with_no_source_and_unlinked_source(
        self, subjects, subjectsources, sources
    ):
        issues = get_subject_source_issues(subjects, subjectsources, sources)
        assert issues[["Issue", "Name"]].values.tolist() == [
            [NO_SOURCE, "Ranger Truck 3"],
            [NO_SUBJECT, "GSM-3001"],
        ]
        no_source = _issues(issues, NO_SOURCE).iloc[0]
        assert no_source["Record type"] == "Subject"
        assert no_source["ID"] == "truck"
        assert no_source["Detail"] == "vehicle, car"


class TestNoAssignments:
    def test_every_subject_and_source_flagged(self, subjects, sources):
        empty = pd.DataFrame(columns=["id", "subject", "source", "assigned_range"])
        issues = get_subject_source_issues(subjects, empty, sources)
        assert len(_issues(issues, NO_SOURCE)) == 4
        assert len(_issues(issues, NO_SUBJECT)) == 5


class TestEmptySite:
    def test_no_rows_but_columns_kept(self):
        issues = get_subject_source_issues(
            pd.DataFrame(columns=["id", "name"]),
            pd.DataFrame(columns=["id", "subject", "source"]),
            pd.DataFrame(columns=["id", "manufacturer_id"]),
        )
        assert issues.empty
        assert list(issues.columns) == ISSUE_COLUMNS


class TestBadDates:
    def test_raw_er_shape(self, subjects, sources):
        links = pd.DataFrame(
            [
                _link("b1", "baraka", "st1002", "0023-06-10T00:00:00+00:00", None),
                _link(
                    "z1",
                    "zawadi",
                    "awt2001",
                    "2021-05-01T00:00:00+00:00",
                    "1022-11-30T00:00:00+00:00",
                ),
            ]
        )
        bad = _issues(get_subject_source_issues(subjects, links, sources), BAD_DATE)
        assert bad[["Name", "ID", "Detail"]].values.tolist() == [
            ["Baraka → ST-1002", "b1", "Start date 0023-06-10 is out of range"],
            ["Zawadi → AWT-2001", "z1", "End date 1022-11-30 is out of range"],
        ]
        assert set(bad["Record type"]) == {"Assignment"}

    def test_flattened_shape_from_get_subjectsources(self, subjects, sources):
        links = pd.DataFrame(
            [
                {
                    "id": "b1",
                    "subject": "baraka",
                    "source": "st1002",
                    "assigned_range_lower": pd.NaT,
                    "assigned_range_upper": pd.NaT,
                    "invalid_assigned_range_lower": "0023-06-10T00:00:00+00:00",
                    "invalid_assigned_range_upper": None,
                }
            ]
        )
        bad = _issues(get_subject_source_issues(subjects, links, sources), BAD_DATE)
        assert bad["Detail"].tolist() == ["Start date 0023-06-10 is out of range"]

    def test_er_placeholders_are_not_flagged(self, subjects, sources):
        links = pd.DataFrame(
            [_link("a1", "amani", "st1001", "0001-01-01T00:00:00+00:00", OPEN_END)]
        )
        assert _issues(
            get_subject_source_issues(subjects, links, sources), BAD_DATE
        ).empty

    def test_bad_dates_listed_first(self, subjects, sources):
        links = pd.DataFrame(
            [_link("b1", "baraka", "st1002", "0023-06-10T00:00:00+00:00", None)]
        )
        issues = get_subject_source_issues(subjects, links, sources)
        assert issues.iloc[0]["Issue"] == BAD_DATE


class TestOverlaps:
    def test_same_source_on_two_subjects_at_once(self, subjects, sources):
        links = pd.DataFrame(
            [
                _link("a1", "amani", "st1001", "2024-03-01T00:00:00+00:00", OPEN_END),
                _link(
                    "b1",
                    "baraka",
                    "st1001",
                    "2024-06-01T00:00:00+00:00",
                    "2024-07-01T00:00:00+00:00",
                ),
            ]
        )
        overlap = _issues(get_subject_source_issues(subjects, links, sources), OVERLAP)
        assert overlap[["Record type", "Name", "ID"]].values.tolist() == [
            ["Source", "ST-1001", "st1001"]
        ]
        assert "Amani (2024-03-01 → present)" in overlap.iloc[0]["Detail"]
        assert "Baraka (2024-06-01 → 2024-07-01)" in overlap.iloc[0]["Detail"]

    def test_hand_over_on_same_day_is_not_an_overlap(self, subjects, sources):
        links = pd.DataFrame(
            [
                _link(
                    "a1",
                    "amani",
                    "st1001",
                    "2024-01-01T00:00:00+00:00",
                    "2024-03-07T00:00:00+00:00",
                ),
                _link("b1", "baraka", "st1001", "2024-03-07T00:00:00+00:00", OPEN_END),
            ]
        )
        assert _issues(
            get_subject_source_issues(subjects, links, sources), OVERLAP
        ).empty

    def test_open_start_counts_as_unbounded(self, subjects, sources):
        links = pd.DataFrame(
            [
                _link("a1", "amani", "st1001", "0001-01-01T00:00:00+00:00", OPEN_END),
                _link("b1", "baraka", "st1001", "2024-03-07T00:00:00+00:00", OPEN_END),
            ]
        )
        assert (
            len(_issues(get_subject_source_issues(subjects, links, sources), OVERLAP))
            == 1
        )

    def test_same_subject_reassigned_is_not_an_overlap(self, subjects, sources):
        links = pd.DataFrame(
            [
                _link("a1", "amani", "st1001", "2024-01-01T00:00:00+00:00", OPEN_END),
                _link("a2", "amani", "st1001", "2024-02-01T00:00:00+00:00", OPEN_END),
            ]
        )
        assert _issues(
            get_subject_source_issues(subjects, links, sources), OVERLAP
        ).empty

    def test_can_be_turned_off(self, subjects, sources):
        links = pd.DataFrame(
            [
                _link("a1", "amani", "st1001", "2024-03-01T00:00:00+00:00", OPEN_END),
                _link("b1", "baraka", "st1001", "2024-06-01T00:00:00+00:00", OPEN_END),
            ]
        )
        issues = get_subject_source_issues(
            subjects, links, sources, check_overlaps=False
        )
        assert _issues(issues, OVERLAP).empty


class TestOtherChecks:
    def test_assignment_to_missing_source(self, subjects, sources):
        links = pd.DataFrame(
            [_link("a1", "amani", "ghost", "2024-03-01T00:00:00+00:00", OPEN_END)]
        )
        missing = _issues(
            get_subject_source_issues(subjects, links, sources), MISSING_SOURCE
        )
        assert missing[["Name", "ID", "Detail"]].values.tolist() == [
            ["Amani → ghost", "a1", "Source ghost is not in the sources list"]
        ]

    def test_source_only_on_excluded_subject_says_why(
        self, subjects, subjectsources, sources
    ):
        active_only = subjects[subjects["is_active"]]
        no_subject = _issues(
            get_subject_source_issues(active_only, subjectsources, sources), NO_SUBJECT
        )
        awt = no_subject[no_subject["Name"] == "AWT-2001"].iloc[0]
        assert (
            awt["Detail"]
            == "awt. Only assigned to subjects not in this report (e.g. inactive)"
        )

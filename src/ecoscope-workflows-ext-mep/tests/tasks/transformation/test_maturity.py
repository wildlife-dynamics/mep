"""Tests for ecoscope_workflows_ext_mep.tasks.transformation._maturity.

`compute_subject_maturity` is registered via `wt_registry.register()`, a
no-op decorator at call time -- including for its pandera-typed
`DataFrame[RelocationsGDFSchema]` parameter, which is *not* validated
just by calling the function directly. So it is exercised here as plain
Python against small, hand-built DataFrames.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_mep.tasks.transformation._maturity import (
    compute_subject_maturity,
)


def _relocs(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal relocations frame from (groupby_col, fixtime) pairs."""
    groupby_col, fixtime = zip(*rows) if rows else ([], [])
    return pd.DataFrame({"groupby_col": groupby_col, "fixtime": fixtime})


class TestComputeSubjectMaturity:
    def test_marks_subject_mature_when_span_meets_duration(self):
        subjects_df = pd.DataFrame({"groupby_col": ["s1"], "subject_name": ["Cherop"]})
        relocations_gdf = _relocs(
            [
                ("s1", "2024-01-01"),
                ("s1", "2024-08-01"),  # 7 months later
            ]
        )

        result = compute_subject_maturity(
            subjects_df=subjects_df, relocations_gdf=relocations_gdf, months_duration=6
        )

        assert bool(result.loc[result["groupby_col"] == "s1", "mature"].iloc[0]) is True

    def test_marks_subject_immature_when_span_is_short(self):
        subjects_df = pd.DataFrame({"groupby_col": ["s1"], "subject_name": ["Cherop"]})
        relocations_gdf = _relocs(
            [
                ("s1", "2024-01-01"),
                ("s1", "2024-02-01"),  # 1 month later
            ]
        )

        result = compute_subject_maturity(
            subjects_df=subjects_df, relocations_gdf=relocations_gdf, months_duration=6
        )

        assert bool(result.loc[result["groupby_col"] == "s1", "mature"].iloc[0]) is False

    def test_subject_with_no_relocations_defaults_to_false(self):
        subjects_df = pd.DataFrame({"groupby_col": ["s1", "s2"], "subject_name": ["Cherop", "Esposito"]})
        relocations_gdf = _relocs([("s1", "2024-01-01"), ("s1", "2024-08-01")])

        result = compute_subject_maturity(subjects_df=subjects_df, relocations_gdf=relocations_gdf)

        s2_mature = result.loc[result["groupby_col"] == "s2", "mature"].iloc[0]
        assert bool(s2_mature) is False

    def test_unparsable_fixtime_rows_are_dropped_before_span_calculation(self):
        subjects_df = pd.DataFrame({"groupby_col": ["s1"], "subject_name": ["Cherop"]})
        relocations_gdf = _relocs(
            [
                ("s1", "2024-01-01"),
                ("s1", "not-a-date"),
                ("s1", "2024-08-01"),
            ]
        )

        result = compute_subject_maturity(
            subjects_df=subjects_df, relocations_gdf=relocations_gdf, months_duration=6
        )

        assert bool(result.loc[result["groupby_col"] == "s1", "mature"].iloc[0]) is True

    def test_custom_months_duration_changes_the_threshold(self):
        subjects_df = pd.DataFrame({"groupby_col": ["s1"], "subject_name": ["Cherop"]})
        relocations_gdf = _relocs([("s1", "2024-01-01"), ("s1", "2024-08-01")])  # 7 months

        mature_at_6mo = compute_subject_maturity(
            subjects_df=subjects_df, relocations_gdf=relocations_gdf, months_duration=6
        )
        mature_at_12mo = compute_subject_maturity(
            subjects_df=subjects_df, relocations_gdf=relocations_gdf, months_duration=12
        )

        assert bool(mature_at_6mo.loc[0, "mature"]) is True
        assert bool(mature_at_12mo.loc[0, "mature"]) is False

    def test_does_not_mutate_input_subjects_df_columns(self):
        subjects_df = pd.DataFrame({"groupby_col": ["s1"], "subject_name": ["Cherop"]})
        original_columns = list(subjects_df.columns)
        relocations_gdf = _relocs([("s1", "2024-01-01"), ("s1", "2024-08-01")])

        compute_subject_maturity(subjects_df=subjects_df, relocations_gdf=relocations_gdf)

        assert list(subjects_df.columns) == original_columns

    def test_missing_groupby_col_in_subjects_raises(self):
        subjects_df = pd.DataFrame({"subject_name": ["Cherop"]})
        relocations_gdf = _relocs([("s1", "2024-01-01"), ("s1", "2024-08-01")])

        with pytest.raises(KeyError):
            compute_subject_maturity(subjects_df=subjects_df, relocations_gdf=relocations_gdf)

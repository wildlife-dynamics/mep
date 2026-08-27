"""Tests for ecoscope_workflows_ext_mep.tasks.results._info.

`process_subject_information` is registered via `wt_registry.register()`,
a no-op decorator at call time, so it is called directly as plain Python.
The module-level helpers (`safe_strip`, `scalar`, `truncate_at_sentence`,
`format_date`) are not registered but are exercised directly too, since
they carry most of the interesting edge-case behavior.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ecoscope_workflows_ext_mep.tasks.results._info import (
    OUTPUT_COLUMNS,
    format_date,
    process_subject_information,
    safe_strip,
    scalar,
    truncate_at_sentence,
)


# --------------------------------------------------------------------------- #
# safe_strip                                                                  #
# --------------------------------------------------------------------------- #


class TestSafeStrip:
    def test_none_returns_empty_string(self):
        assert safe_strip(None) == ""

    def test_nan_float_returns_empty_string(self):
        assert safe_strip(math.nan) == ""

    def test_strips_whitespace(self):
        assert safe_strip("  Cherop  ") == "Cherop"

    @pytest.mark.parametrize("value", ["nan", "NaN", "none", "None", "nat", "NaT"])
    def test_literal_null_like_strings_are_treated_as_empty(self, value):
        assert safe_strip(value) == ""

    def test_non_string_scalar_is_stringified(self):
        assert safe_strip(1985) == "1985"


# --------------------------------------------------------------------------- #
# scalar                                                                      #
# --------------------------------------------------------------------------- #


class TestScalar:
    def test_returns_value_for_normal_column(self):
        row = pd.Series({"subject_name": "Cherop"})
        assert scalar(row, "subject_name") == "Cherop"

    def test_missing_key_returns_default(self):
        row = pd.Series({"subject_name": "Cherop"})
        assert scalar(row, "missing_key", default="fallback") == "fallback"

    def test_duplicate_columns_returns_first_non_null(self):
        # Simulate row[key] being a Series, as happens with duplicate column
        # names in the source DataFrame.
        row = pd.Series({"subject_name": "Cherop"})
        dup_value = pd.Series([None, "Cherop", "OtherName"])
        row["subject_name"] = dup_value

        result = scalar(row, "subject_name")

        assert result == "Cherop"

    def test_duplicate_columns_all_null_returns_default(self):
        row = pd.Series({"subject_name": "placeholder"})
        row["subject_name"] = pd.Series([None, None])

        result = scalar(row, "subject_name", default="fallback")

        assert result == "fallback"


# --------------------------------------------------------------------------- #
# truncate_at_sentence                                                        #
# --------------------------------------------------------------------------- #


class TestTruncateAtSentence:
    def test_short_text_is_unchanged(self):
        assert truncate_at_sentence("Short bio.", maxlen=100) == "Short bio."

    def test_cuts_at_last_sentence_boundary_past_min_pos(self):
        text = "A" * 45 + ". " + "This part gets cut off entirely here."
        result = truncate_at_sentence(text, maxlen=50, min_sentence_pos=40)
        assert result.endswith(".")
        assert len(result) <= 50

    def test_falls_back_to_ellipsis_when_no_sentence_boundary(self):
        text = "A" * 100
        result = truncate_at_sentence(text, maxlen=50, min_sentence_pos=40)
        assert result.endswith("...")

    def test_exact_maxlen_boundary_is_not_truncated(self):
        text = "A" * 50
        assert truncate_at_sentence(text, maxlen=50) == text


# --------------------------------------------------------------------------- #
# format_date                                                                 #
# --------------------------------------------------------------------------- #


class TestFormatDate:
    def test_empty_or_null_like_input_returns_empty_string(self):
        assert format_date(None) == ""
        assert format_date("") == ""
        assert format_date("nan") == ""

    def test_year_only_float_returns_bare_year(self):
        assert format_date(1985.0) == "1985"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2024-03-05", "05 Mar 2024"),
            ("05/03/2024", "05 Mar 2024"),
            ("05-03-2024", "05 Mar 2024"),
            ("Mar 05, 2024", "05 Mar 2024"),
            ("05 Mar 2024", "05 Mar 2024"),
            ("2024/03/05", "05 Mar 2024"),
        ],
    )
    def test_parses_known_formats(self, raw, expected):
        assert format_date(raw) == expected

    def test_unparsable_string_is_returned_unchanged(self):
        assert format_date("not-a-date-at-all") == "not-a-date-at-all"


# --------------------------------------------------------------------------- #
# process_subject_information                                                 #
# --------------------------------------------------------------------------- #


class TestProcessSubjectInformation:
    def test_empty_dataframe_returns_empty_with_output_columns(self):
        result = process_subject_information(pd.DataFrame())

        assert result.empty
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_maps_and_formats_a_single_subject_row(self):
        subject_df = pd.DataFrame(
            [
                {
                    "subject_name": "cherop",
                    "date_of_birth": "2020-05-01",
                    "subject_sex": "female",
                    "country": "Kenya",
                    "notes": "  some notes  ",
                    "status": "Active",
                    "subject_bio": "A short bio.",
                    "distribution": "Amboseli",
                }
            ]
        )

        result = process_subject_information(subject_df)

        row = result.iloc[0]
        assert row["subject_name"] == "Cherop"
        assert row["dob"] == "01 May 2020"
        assert row["sex"] == "Female"
        assert row["country"] == "Kenya"
        assert row["notes"] == "some notes"
        assert row["status_raw"] == "Active"
        assert row["bio"] == "A short bio."
        assert row["distribution"] == "Amboseli"

    def test_active_status_is_rendered_green(self):
        subject_df = pd.DataFrame([{"status": "active"}])

        result = process_subject_information(subject_df)

        assert "color: green" in result.iloc[0]["status"]

    def test_inactive_status_is_rendered_red(self):
        subject_df = pd.DataFrame([{"status": "deceased"}])

        result = process_subject_information(subject_df)

        assert "color: red" in result.iloc[0]["status"]

    def test_missing_status_produces_empty_status_html(self):
        subject_df = pd.DataFrame([{"subject_name": "cherop"}])

        result = process_subject_information(subject_df)

        assert result.iloc[0]["status"] == ""

    def test_status_value_is_html_escaped(self):
        subject_df = pd.DataFrame([{"status": "<script>alert(1)</script>"}])

        result = process_subject_information(subject_df)

        assert "<script>" not in result.iloc[0]["status"]
        assert "&lt;script&gt;" in result.iloc[0]["status"]

    def test_bio_is_truncated_using_maxlen(self):
        # No sentence boundary exists in an all-"A" string, so
        # `truncate_at_sentence` falls back to `cut + "..."`, which can be
        # a few characters *longer* than maxlen -- not a hard cap.
        subject_df = pd.DataFrame([{"subject_bio": "A" * 200}])

        result = process_subject_information(subject_df, maxlen=50)

        bio = result.iloc[0]["bio"]
        assert bio.endswith("...")
        assert len(bio) <= 53

    def test_missing_optional_columns_default_to_empty_strings(self):
        subject_df = pd.DataFrame([{"subject_name": "cherop"}])

        result = process_subject_information(subject_df)

        row = result.iloc[0]
        assert row["dob"] == ""
        assert row["country"] == ""
        assert row["notes"] == ""
        assert row["bio"] == ""
        assert row["distribution"] == ""

    def test_output_has_exactly_output_columns_in_order(self):
        subject_df = pd.DataFrame([{"subject_name": "cherop"}])

        result = process_subject_information(subject_df)

        assert list(result.columns) == OUTPUT_COLUMNS

    def test_multiple_rows_are_each_processed(self):
        subject_df = pd.DataFrame([{"subject_name": "cherop"}, {"subject_name": "esposito"}])

        result = process_subject_information(subject_df)

        assert list(result["subject_name"]) == ["Cherop", "Esposito"]

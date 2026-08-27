"""Tests for ecoscope_workflows_ext_mep.tasks.reporting._subject_tracking.

`generate_subject_report` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python
against a real .docx template (built via python-docx) and real CSV/PNG
asset files on disk (via `tmp_path`), rather than mocking `docxtpl` or the
filesystem.

The module's non-registered helpers (`resolve_subject_file_paths`,
`_safe_get`, `_read_csv`, `is_valid_image`, `build_subject_context`,
`prepare_context`) are exercised directly too, since most of the
interesting edge-case behavior (missing files, invalid images, NaN
values) lives there.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from docxtpl import DocxTemplate, InlineImage

from ecoscope_workflows_ext_mep.tasks.reporting._subject_tracking import (
    IMAGE_FIELD_MAPPING,
    _read_csv,
    _safe_get,
    build_subject_context,
    create_inline_image_inch,
    generate_subject_report,
    is_valid_image,
    prepare_context,
    resolve_subject_file_paths,
)

VALID_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
VALID_JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 16


def _write_bytes(path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# --------------------------------------------------------------------------- #
# resolve_subject_file_paths                                                  #
# --------------------------------------------------------------------------- #


class TestResolveSubjectFilePaths:
    def test_existing_files_resolve_to_their_path(self, tmp_path):
        _write_bytes(tmp_path / "Cherop_speedmap.png", VALID_PNG_BYTES)

        result = resolve_subject_file_paths(str(tmp_path), "Cherop")

        assert result["speedmap_path"] == str(tmp_path / "Cherop_speedmap.png")

    def test_missing_files_resolve_to_none(self, tmp_path):
        result = resolve_subject_file_paths(str(tmp_path), "Cherop")

        assert all(v is None for v in result.values())

    def test_covers_all_eleven_expected_keys(self, tmp_path):
        result = resolve_subject_file_paths(str(tmp_path), "Cherop")

        assert set(result.keys()) == {
            "profile_photo_path",
            "subject_info_path",
            "speedmap_path",
            "homerange_map_path",
            "seasonal_homerange_map_path",
            "nsd_plot_path",
            "speed_plot_path",
            "collared_event_plot_path",
            "mcp_plot_path",
            "subject_stats_table_path",
            "subject_occupancy_table_path",
        }


# --------------------------------------------------------------------------- #
# _safe_get                                                                   #
# --------------------------------------------------------------------------- #


class TestSafeGet:
    def test_none_df_returns_default(self):
        assert _safe_get(None, "col") == "Undefined"

    def test_empty_df_returns_default(self):
        assert _safe_get(pd.DataFrame(), "col") == "Undefined"

    def test_missing_column_returns_default(self):
        df = pd.DataFrame({"other": [1]})
        assert _safe_get(df, "col", default="fallback") == "fallback"

    def test_nan_value_returns_default(self):
        df = pd.DataFrame({"col": [math.nan]})
        assert _safe_get(df, "col") == "Undefined"

    def test_present_value_is_returned(self):
        df = pd.DataFrame({"col": ["hello"]})
        assert _safe_get(df, "col") == "hello"


# --------------------------------------------------------------------------- #
# _read_csv                                                                   #
# --------------------------------------------------------------------------- #


class TestReadCsv:
    def test_none_path_returns_none(self):
        assert _read_csv(None) is None

    def test_nonexistent_path_returns_none(self, tmp_path):
        assert _read_csv(str(tmp_path / "missing.csv")) is None

    def test_existing_csv_is_read(self, tmp_path):
        path = tmp_path / "info.csv"
        pd.DataFrame({"subject_name": ["Cherop"]}).to_csv(path, index=False)

        result = _read_csv(str(path))

        assert result is not None
        assert result.iloc[0]["subject_name"] == "Cherop"


# --------------------------------------------------------------------------- #
# is_valid_image                                                              #
# --------------------------------------------------------------------------- #


class TestIsValidImage:
    def test_valid_png_header_is_true(self, tmp_path):
        path = tmp_path / "a.png"
        _write_bytes(path, VALID_PNG_BYTES)

        assert is_valid_image(str(path)) is True

    def test_valid_jpeg_header_is_true(self, tmp_path):
        path = tmp_path / "a.jpg"
        _write_bytes(path, VALID_JPEG_BYTES)

        assert is_valid_image(str(path)) is True

    def test_invalid_header_is_false(self, tmp_path):
        path = tmp_path / "a.png"
        _write_bytes(path, b"not-an-image")

        assert is_valid_image(str(path)) is False

    def test_missing_file_is_false(self, tmp_path):
        assert is_valid_image(str(tmp_path / "missing.png")) is False


# --------------------------------------------------------------------------- #
# create_inline_image_inch                                                    #
# --------------------------------------------------------------------------- #


class TestCreateInlineImageInch:
    def test_returns_an_inline_image(self, tmp_path, make_png, make_docx_template):
        png_path = make_png(tmp_path / "photo.png")
        template_path = make_docx_template(["{{ photo }}"])
        template = DocxTemplate(str(template_path))

        result = create_inline_image_inch(template, str(png_path), width_cm=2.0, height_cm=3.0)

        assert isinstance(result, InlineImage)


# --------------------------------------------------------------------------- #
# build_subject_context                                                       #
# --------------------------------------------------------------------------- #


class TestBuildSubjectContext:
    def test_reads_csvs_and_resolves_media_paths(self, tmp_path, make_png):
        subject_name = "Cherop"
        pd.DataFrame(
            [{"subject_name": subject_name, "dob": "01 Jan 2020", "sex": "Female", "country": "Kenya"}]
        ).to_csv(tmp_path / f"{subject_name}_subject_info.csv", index=False)
        pd.DataFrame([{"mcp": 12.3, "etd": 4.5}]).to_csv(
            tmp_path / f"{subject_name}_subject_stats.csv", index=False
        )
        pd.DataFrame([{"national_pa_use": 50.0}]).to_csv(
            tmp_path / f"{subject_name}_subject_occupancy.csv", index=False
        )
        make_png(tmp_path / f"{subject_name}.png")

        df = pd.DataFrame({"subject_name": [subject_name]})
        context = build_subject_context(df=df, output_dir=str(tmp_path))

        assert context["name"] == subject_name
        assert context["dob"] == "01 Jan 2020"
        assert context["sex"] == "Female"
        assert context["mcp"] == 12.3
        assert context["national_pa_use"] == 50.0
        assert context["profile_photo"] == str(tmp_path / f"{subject_name}.png")

    def test_missing_assets_fall_back_to_defaults(self, tmp_path):
        df = pd.DataFrame({"subject_name": ["Ghost"]})

        context = build_subject_context(df=df, output_dir=str(tmp_path))

        assert context["name"] == "Undefined"
        assert context["mcp"] == 0
        assert context["national_pa_use"] == 0
        assert context["profile_photo"] is None


# --------------------------------------------------------------------------- #
# prepare_context                                                             #
# --------------------------------------------------------------------------- #


class TestPrepareContext:
    def test_valid_image_path_becomes_inline_image(self, tmp_path, make_png, make_docx_template):
        png_path = make_png(tmp_path / "photo.png")
        template = DocxTemplate(str(make_docx_template(["{{ profile_photo }}"])))
        context = {"profile_photo": str(png_path)}

        result = prepare_context(context, template)

        assert isinstance(result["profile_photo"], InlineImage)

    def test_missing_image_path_becomes_none(self, tmp_path, make_docx_template):
        template = DocxTemplate(str(make_docx_template(["{{ profile_photo }}"])))
        context = {"profile_photo": str(tmp_path / "missing.png")}

        result = prepare_context(context, template)

        assert result["profile_photo"] is None

    def test_invalid_image_bytes_become_none(self, tmp_path, make_docx_template):
        bad_path = tmp_path / "bad.png"
        _write_bytes(bad_path, b"not-an-image")
        template = DocxTemplate(str(make_docx_template(["{{ profile_photo }}"])))
        context = {"profile_photo": str(bad_path)}

        result = prepare_context(context, template)

        assert result["profile_photo"] is None

    def test_empty_image_path_becomes_none(self, tmp_path, make_docx_template):
        template = DocxTemplate(str(make_docx_template(["{{ profile_photo }}"])))
        context = {"profile_photo": ""}

        result = prepare_context(context, template)

        assert result["profile_photo"] is None

    def test_non_image_fields_are_passed_through_unchanged(self, tmp_path, make_docx_template):
        template = DocxTemplate(str(make_docx_template(["{{ name }}"])))
        context = {"name": "Cherop"}

        result = prepare_context(context, template)

        assert result["name"] == "Cherop"

    def test_fields_not_in_image_mapping_are_untouched_even_if_set(self, make_docx_template):
        template = DocxTemplate(str(make_docx_template(["{{ x }}"])))
        context = {field: "sentinel" for field in IMAGE_FIELD_MAPPING}
        context["unrelated_field"] = "sentinel"

        result = prepare_context(context, template)

        assert result["unrelated_field"] == "sentinel"


# --------------------------------------------------------------------------- #
# generate_subject_report                                                     #
# --------------------------------------------------------------------------- #


class TestGenerateSubjectReport:
    def test_renders_and_saves_a_docx_with_subject_fields(
        self, tmp_path, make_png, make_docx_template, read_docx_text
    ):
        subject_name = "Cherop"
        pd.DataFrame([{"subject_name": subject_name, "dob": "01 Jan 2020"}]).to_csv(
            tmp_path / f"{subject_name}_subject_info.csv", index=False
        )
        make_png(tmp_path / f"{subject_name}.png")
        template_path = make_docx_template(["Name: {{ name }}", "DOB: {{ dob }}"])

        df = pd.DataFrame({"subject_name": [subject_name]})

        result_path = generate_subject_report(
            df=df, output_dir=str(tmp_path), template_path=str(template_path)
        )

        assert result_path == str(tmp_path / "cherop.docx")
        texts = read_docx_text(result_path)
        assert "Name: Cherop" in texts
        assert "DOB: 01 Jan 2020" in texts

    def test_output_filename_is_sanitized_via_safe_string(self, tmp_path, make_docx_template):
        subject_name = "Cherop The Elephant!"
        df = pd.DataFrame({"subject_name": [subject_name]})
        template_path = make_docx_template(["{{ name }}"])

        result_path = generate_subject_report(
            df=df, output_dir=str(tmp_path), template_path=str(template_path)
        )

        assert result_path == str(tmp_path / "cherop_the_elephant.docx")

    def test_missing_assets_still_produce_a_document(self, tmp_path, make_docx_template):
        df = pd.DataFrame({"subject_name": ["Ghost"]})
        template_path = make_docx_template(["{{ name }}"])

        result_path = generate_subject_report(
            df=df, output_dir=str(tmp_path), template_path=str(template_path)
        )

        assert result_path is not None
        from pathlib import Path

        assert Path(result_path).exists()

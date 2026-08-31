"""Tests for ecoscope_workflows_ext_mep.tasks.reporting._monthly_report.

`create_mep_monthly_report` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python. It
classifies chart images and the sitrep CSV by walking `output_dir` and
matching the literal filename stems produced upstream by the
mep-monthly_report workflow spec (`elephant_speedmap`,
`elephant_sightings_map`, `vehicle_patrols_map`, `foot_patrols_map`,
`{subject}_historic_voltage`, `sitrep_report`), so tests exercise that
classification against real files on disk (via `tmp_path`) and a real
.docx template (via python-docx), rather than mocking the filesystem or
docxtpl.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from docxtpl import DocxTemplate, InlineImage

from ecoscope_workflows_ext_mep.tasks.reporting._monthly_report import (
    _maybe_image,
    _read_csv,
    create_mep_monthly_report,
)


def _report_template(make_docx_template) -> Path:
    return make_docx_template(
        [
            "{% if elephant_speedmap %}HAS_SPEEDMAP{% endif %}",
            "{% if elephant_sighting_map %}HAS_SIGHTINGS{% endif %}",
            "{% if vehicle_patrol_tracks %}HAS_VEHICLE{% endif %}",
            "{% if foot_patrol_tracks %}HAS_FOOT{% endif %}",
            "{% for cv in collar_voltage_list %}VOLTAGE:{{ cv.subject }} {% endfor %}",
            "{% for row in sitrep %}SITREP:{{ row.name }} {% endfor %}",
        ]
    )


# --------------------------------------------------------------------------- #
# _read_csv                                                                   #
# --------------------------------------------------------------------------- #


class TestReadCsv:
    def test_none_path_returns_none(self):
        assert _read_csv(None) is None

    def test_nonexistent_path_returns_none(self, tmp_path):
        assert _read_csv(str(tmp_path / "missing.csv")) is None

    def test_existing_csv_is_read(self, tmp_path):
        path = tmp_path / "sitrep_report.csv"
        pd.DataFrame({"name": ["Arrest near Loita"]}).to_csv(path, index=False)

        result = _read_csv(str(path))

        assert result is not None
        assert result.iloc[0]["name"] == "Arrest near Loita"


# --------------------------------------------------------------------------- #
# _maybe_image                                                                #
# --------------------------------------------------------------------------- #


class TestMaybeImage:
    def test_none_path_returns_none(self, make_docx_template):
        tpl = DocxTemplate(str(make_docx_template(["{{ x }}"])))

        assert _maybe_image(tpl, None) is None

    def test_empty_path_returns_none(self, make_docx_template):
        tpl = DocxTemplate(str(make_docx_template(["{{ x }}"])))

        assert _maybe_image(tpl, "") is None

    def test_real_path_returns_inline_image(self, tmp_path, make_png, make_docx_template):
        png_path = make_png(tmp_path / "chart.png")
        tpl = DocxTemplate(str(make_docx_template(["{{ x }}"])))

        result = _maybe_image(tpl, str(png_path))

        assert isinstance(result, InlineImage)


# --------------------------------------------------------------------------- #
# create_mep_monthly_report                                                   #
# --------------------------------------------------------------------------- #


class TestCreateMepMonthlyReport:
    def test_classifies_each_chart_by_its_known_filename(self, tmp_path, make_png, make_docx_template, read_docx_text):
        make_png(tmp_path / "elephant_speedmap.png")
        make_png(tmp_path / "elephant_sightings_map.png")
        make_png(tmp_path / "vehicle_patrols_map.png")
        make_png(tmp_path / "foot_patrols_map.png")
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "HAS_SPEEDMAP" in texts
        assert "HAS_SIGHTINGS" in texts
        assert "HAS_VEHICLE" in texts
        assert "HAS_FOOT" in texts

    def test_unrelated_images_are_ignored(self, tmp_path, make_png, make_docx_template, read_docx_text):
        make_png(tmp_path / "some_other_chart.png")
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "HAS_SPEEDMAP" not in texts
        assert "HAS_SIGHTINGS" not in texts

    def test_voltage_charts_are_grouped_by_subject_with_suffix_stripped(
        self, tmp_path, make_png, make_docx_template, read_docx_text
    ):
        make_png(tmp_path / "Cherop_historic_voltage.png")
        make_png(tmp_path / "Tembo_historic_voltage.png")
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "VOLTAGE:Cherop" in texts
        assert "VOLTAGE:Tembo" in texts

    def test_sitrep_csv_is_found_and_rendered_by_matching_stem(self, tmp_path, make_docx_template, read_docx_text):
        pd.DataFrame({"name": ["Arrest near Loita"]}).to_csv(tmp_path / "sitrep_report.csv", index=False)
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "SITREP:Arrest near Loita" in texts

    def test_other_csvs_are_not_mistaken_for_the_sitrep(self, tmp_path, make_docx_template, read_docx_text):
        pd.DataFrame({"name": ["not sitrep data"]}).to_csv(tmp_path / "some_other_table.csv", index=False)
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "SITREP:" not in texts

    def test_charts_found_in_nested_subdirectories_are_still_picked_up(
        self, tmp_path, make_png, make_docx_template, read_docx_text
    ):
        make_png(tmp_path / "subdir" / "elephant_speedmap.png")
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(tmp_path))

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "HAS_SPEEDMAP" in texts

    def test_no_charts_or_sitrep_still_produces_a_document(self, tmp_path, make_docx_template):
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(tmp_path))

        assert Path(result_path).exists()

    def test_default_filename_is_timestamped_docx(self, tmp_path, make_docx_template):
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(tmp_path))

        assert Path(result_path).parent == tmp_path
        assert Path(result_path).name.startswith("mep_monthly_report_")
        assert Path(result_path).suffix == ".docx"

    def test_explicit_filename_is_used_verbatim(self, tmp_path, make_docx_template):
        template_path = _report_template(make_docx_template)

        result_path = create_mep_monthly_report(
            template_path=str(template_path),
            output_dir=str(tmp_path),
            filename="custom_report.docx",
        )

        assert Path(result_path) == tmp_path / "custom_report.docx"

    def test_output_dir_is_created_if_it_does_not_exist_yet(self, tmp_path, make_docx_template):
        template_path = _report_template(make_docx_template)
        fresh_output_dir = tmp_path / "results" / "nested"

        result_path = create_mep_monthly_report(template_path=str(template_path), output_dir=str(fresh_output_dir))

        assert Path(result_path).exists()

    def test_invalid_template_path_raises_value_error(self, tmp_path):
        # docxtpl's DocxTemplate doesn't validate the package eagerly on
        # construction, so a missing template surfaces as a ValueError from
        # either the "load template" or the later "render or save" branch.
        with pytest.raises(ValueError, match="missing_template.docx"):
            create_mep_monthly_report(
                template_path=str(tmp_path / "missing_template.docx"),
                output_dir=str(tmp_path),
            )

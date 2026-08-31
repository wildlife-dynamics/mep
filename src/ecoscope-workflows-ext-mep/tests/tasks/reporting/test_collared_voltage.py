"""Tests for ecoscope_workflows_ext_mep.tasks.reporting._collared_voltage.

`generate_source_voltage_report` is registered via `wt_registry.register()`,
a no-op decorator at call time, so it is called directly as plain Python
against a real .docx template (via python-docx) and real PNG files on disk
(via `tmp_path`/Pillow), rather than mocking docxtpl or the org-logo
dimension calculation (`get_image_dimensions_from_pixels`, which is cheap
Pillow code and installed in this environment).

Unlike `create_mep_monthly_report`, this task doesn't restrict itself to a
known set of filename stems: every image found under `output_dir` becomes
a voltage-chart entry, with `_historic_voltage`-suffixed files grouped by
subject (suffix stripped) and any other image using its full stem as the
"subject" label.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from ecoscope.platform.tasks.filter._filter import TimeRange, TimezoneInfo

from ecoscope_workflows_ext_mep.tasks.reporting._collared_voltage import generate_source_voltage_report


@pytest.fixture
def report_period() -> TimeRange:
    tz = TimezoneInfo(label="UTC", tzCode="UTC", name="UTC", utc="+00:00")
    return TimeRange(since=datetime(2024, 1, 1), until=datetime(2024, 2, 1), timezone=tz)


def _template(make_docx_template) -> Path:
    return make_docx_template(
        [
            "Prepared by: {{ prepared_by }}",
            "Period: {{ report_period }}",
            "{% if org_logo %}HAS_LOGO{% endif %}",
            "{% for sv in source_voltage_charts %}VOLT:{{ sv.subject }} {% endfor %}",
        ]
    )


class TestGenerateSourceVoltageReport:
    def test_voltage_charts_grouped_by_subject_with_suffix_stripped(
        self, tmp_path, make_png, make_docx_template, read_docx_text, report_period
    ):
        make_png(tmp_path / "Cherop_historic_voltage.png")
        make_png(tmp_path / "Tembo_historic_voltage.png")
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=str(tmp_path),
            template_path=str(template_path),
        )

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "VOLT:Cherop" in texts
        assert "VOLT:Tembo" in texts

    def test_non_suffixed_image_uses_full_stem_as_subject(
        self, tmp_path, make_png, make_docx_template, read_docx_text, report_period
    ):
        make_png(tmp_path / "random_chart.png")
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=str(tmp_path),
            template_path=str(template_path),
        )

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "VOLT:random_chart" in texts

    def test_org_logo_included_when_provided(
        self, tmp_path, make_png, make_docx_template, read_docx_text, report_period
    ):
        logo_path = make_png(tmp_path / "logo.png", size=(200, 100))
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=str(logo_path),
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=str(tmp_path),
            template_path=str(template_path),
        )

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "HAS_LOGO" in texts

    def test_org_logo_omitted_when_none(
        self, tmp_path, make_docx_template, read_docx_text, report_period
    ):
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=str(tmp_path),
            template_path=str(template_path),
        )

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "HAS_LOGO" not in texts

    def test_empty_org_logo_path_raises(self, tmp_path, make_docx_template, report_period):
        template_path = _template(make_docx_template)

        with pytest.raises(ValueError, match="org_logo_path is empty"):
            generate_source_voltage_report(
                org_logo_path="",
                report_period=report_period,
                prepared_by="Wildlife Dynamics",
                output_dir=str(tmp_path),
                template_path=str(template_path),
            )

    def test_prepared_by_and_report_period_are_rendered(
        self, tmp_path, make_docx_template, read_docx_text, report_period
    ):
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=str(tmp_path),
            template_path=str(template_path),
        )

        texts = " ".join(read_docx_text(Path(result_path)))
        assert "Prepared by: Wildlife Dynamics" in texts
        assert "Period: 01 Jan 2024" in texts
        assert "01 Feb 2024" in texts

    def test_default_filename_pattern(self, tmp_path, make_docx_template, report_period):
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=str(tmp_path),
            template_path=str(template_path),
        )

        assert Path(result_path).parent == tmp_path
        assert Path(result_path).name.startswith("source_voltage_report_")
        assert Path(result_path).suffix == ".docx"

    def test_explicit_filename_is_used_verbatim(self, tmp_path, make_docx_template, report_period):
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=str(tmp_path),
            template_path=str(template_path),
            filename="custom_voltage_report.docx",
        )

        assert Path(result_path) == tmp_path / "custom_voltage_report.docx"

    def test_file_scheme_paths_are_stripped(self, tmp_path, make_docx_template, report_period):
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=f"file://{tmp_path}",
            template_path=f"file://{template_path}",
        )

        assert Path(result_path).exists()
        assert not str(result_path).startswith("file://")

    def test_no_charts_found_still_produces_a_document(self, tmp_path, make_docx_template, report_period):
        template_path = _template(make_docx_template)

        result_path = generate_source_voltage_report(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Wildlife Dynamics",
            output_dir=str(tmp_path),
            template_path=str(template_path),
        )

        assert Path(result_path).exists()

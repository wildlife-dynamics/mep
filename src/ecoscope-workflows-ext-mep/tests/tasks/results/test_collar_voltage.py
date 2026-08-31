"""Tests for ecoscope_workflows_ext_mep.tasks.results._collar_voltage.

`plot_historic_voltage` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python
against the real, year-long single-subject relocations fixture (see
`conftest.py`) with a synthetic `voltage` column added, run through the
real `ecoscope.platform` aggregation/plotting machinery -- nothing here
is mocked, since it's all installed and cheap to run for real.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecoscope_workflows_ext_mep.tasks.results._collar_voltage import plot_historic_voltage


@pytest.fixture
def voltage_gdf(relocs_gdf):
    """`relocs_gdf` with a synthetic, mildly-varying `voltage` column."""
    gdf = relocs_gdf.copy()
    rng = np.random.default_rng(0)
    gdf["voltage"] = rng.uniform(3.5, 4.2, size=len(gdf))
    return gdf


class TestPlotHistoricVoltage:
    def test_returns_non_empty_html(self, voltage_gdf):
        html = plot_historic_voltage(current_relocs=voltage_gdf.copy(), previous_relocs=None)

        assert isinstance(html, str)
        assert "<div" in html

    def test_none_previous_relocs_falls_back_to_current(self, voltage_gdf):
        # Should not raise even though there's no prior-run history.
        html = plot_historic_voltage(current_relocs=voltage_gdf.copy(), previous_relocs=None)

        assert "<div" in html

    def test_empty_previous_relocs_falls_back_to_current(self, voltage_gdf):
        empty = voltage_gdf.iloc[0:0]

        html = plot_historic_voltage(current_relocs=voltage_gdf.copy(), previous_relocs=empty)

        assert "<div" in html

    def test_real_previous_relocs_are_used_for_the_historic_band(self, voltage_gdf):
        previous = voltage_gdf.copy()
        previous["voltage"] = previous["voltage"] + 1.0  # distinctly different history

        html = plot_historic_voltage(current_relocs=voltage_gdf.copy(), previous_relocs=previous)

        assert "<div" in html

    def test_constant_voltage_widens_a_collapsed_band_without_raising(self, voltage_gdf):
        constant = voltage_gdf.copy()
        constant["voltage"] = 4.0

        html = plot_historic_voltage(current_relocs=constant.copy(), previous_relocs=None)

        assert "<div" in html

    def test_custom_voltage_column_name_is_respected(self, voltage_gdf):
        renamed = voltage_gdf.rename(columns={"voltage": "batt_volts"})

        html = plot_historic_voltage(current_relocs=renamed.copy(), previous_relocs=None, column="batt_volts")

        assert "<div" in html

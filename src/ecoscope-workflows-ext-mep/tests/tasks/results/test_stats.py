"""Tests for ecoscope_workflows_ext_mep.tasks.results._stats.

`compute_subject_stats` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python
against the real, year-long single-subject trajectory fixture (see
`conftest.py`).

The `night_day_ratio` computation delegates to
`ecoscope.platform.tasks.analysis.summarize_df` with a
`NightDayRatioSummaryParam`. That is `ecoscope.platform` library code with
its own test suite, so `analysis.summarize_df` is monkeypatched out here to
keep the rest of `compute_subject_stats` (MCP, ETD, distance, displacement,
time tracked, maturity lookup) exercised for real.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest

import ecoscope_workflows_ext_mep.tasks.results._stats as stats_mod
from ecoscope_workflows_ext_mep.tasks.results._stats import compute_subject_stats


@pytest.fixture(autouse=True)
def _stub_night_day_ratio(monkeypatch):
    monkeypatch.setattr(
        stats_mod.analysis,
        "summarize_df",
        lambda df, params, groupby_cols=None: pd.DataFrame([{"night_day_ratio": 1.2345}]),
    )


@pytest.fixture
def etd_gdf(traj_gdf) -> gpd.GeoDataFrame:
    point = traj_gdf.geometry.iloc[0].buffer(0.01)
    return gpd.GeoDataFrame(
        {"percentile": [50.0, 99.9], "area_sqkm": [4.5, 12.3]},
        geometry=[point, point],
        crs="EPSG:4326",
    )


class TestComputeSubjectStats:
    def test_computes_stats_from_a_real_year_long_trajectory(self, traj_gdf, etd_gdf):
        subject_id = traj_gdf["groupby_col"].iloc[0]
        subject_df = pd.DataFrame({"groupby_col": [subject_id], "mature": [True]})

        result = compute_subject_stats(traj_gdf=traj_gdf, subject_df=subject_df, etd_df=etd_gdf)

        row = result.iloc[0]
        assert row["subject_id"] == subject_id
        assert bool(row["mature"]) is True
        assert row["time_tracked_days"] == 365
        assert row["time_tracked_years"] == pytest.approx(1.0)
        assert row["etd"] == pytest.approx(12.3)
        assert row["mcp"] > 0
        assert row["distance_travelled"] > 0
        assert row["max_displacement"] > 0
        assert row["night_day_ratio"] == pytest.approx(1.23)

    def test_result_has_exactly_one_row(self, traj_gdf, etd_gdf):
        subject_df = pd.DataFrame({"groupby_col": [traj_gdf["groupby_col"].iloc[0]], "mature": [False]})

        result = compute_subject_stats(traj_gdf=traj_gdf, subject_df=subject_df, etd_df=etd_gdf)

        assert len(result) == 1

    def test_no_subjects_in_groupby_col_raises(self, traj_gdf, etd_gdf):
        empty = traj_gdf.iloc[0:0]

        with pytest.raises(ValueError, match="No non-null values"):
            compute_subject_stats(traj_gdf=empty, subject_df=pd.DataFrame(), etd_df=etd_gdf)

    def test_multiple_subjects_in_groupby_col_raises(self, traj_gdf, etd_gdf):
        mixed = traj_gdf.copy()
        mixed.loc[mixed.index[: len(mixed) // 2], "groupby_col"] = "another-subject"

        with pytest.raises(ValueError, match="Multiple subjects present"):
            compute_subject_stats(traj_gdf=mixed, subject_df=pd.DataFrame(), etd_df=etd_gdf)

    def test_etd_df_none_defaults_to_zero(self, traj_gdf):
        subject_df = pd.DataFrame({"groupby_col": [traj_gdf["groupby_col"].iloc[0]], "mature": [True]})

        result = compute_subject_stats(traj_gdf=traj_gdf, subject_df=subject_df, etd_df=None)

        assert result.iloc[0]["etd"] == 0.0

    def test_etd_df_empty_defaults_to_zero(self, traj_gdf):
        subject_df = pd.DataFrame({"groupby_col": [traj_gdf["groupby_col"].iloc[0]], "mature": [True]})
        empty_etd = gpd.GeoDataFrame({"percentile": [], "area_sqkm": []}, geometry=[], crs="EPSG:4326")

        result = compute_subject_stats(traj_gdf=traj_gdf, subject_df=subject_df, etd_df=empty_etd)

        assert result.iloc[0]["etd"] == 0.0

    def test_missing_dist_meters_column_defaults_distance_to_zero(self, traj_gdf, etd_gdf):
        traj_without_dist = traj_gdf.drop(columns=["dist_meters"])
        subject_df = pd.DataFrame({"groupby_col": [traj_gdf["groupby_col"].iloc[0]], "mature": [True]})

        result = compute_subject_stats(traj_gdf=traj_without_dist, subject_df=subject_df, etd_df=etd_gdf)

        assert result.iloc[0]["distance_travelled"] == 0.0

    def test_subject_not_found_in_subject_df_defaults_mature_to_false(self, traj_gdf, etd_gdf):
        subject_df = pd.DataFrame({"groupby_col": ["some-other-subject"], "mature": [True]})

        result = compute_subject_stats(traj_gdf=traj_gdf, subject_df=subject_df, etd_df=etd_gdf)

        assert bool(result.iloc[0]["mature"]) is False

    def test_empty_subject_df_defaults_mature_to_false(self, traj_gdf, etd_gdf):
        result = compute_subject_stats(traj_gdf=traj_gdf, subject_df=pd.DataFrame(), etd_df=etd_gdf)

        assert bool(result.iloc[0]["mature"]) is False

    def test_dusk_dawn_inside_a_leap_second_still_yields_a_ratio(self, traj_gdf, etd_gdf, monkeypatch):
        # astropy raises by default for a time inside the 2015-06-30 23:59:60 UTC leap second: when
        # ecoscope converts dusk/dawn with Time.to_datetime, and when astroplan's sunrise search turns
        # a datetime-format Time into a string.
        from datetime import datetime

        import astropy.units as u
        import pytz
        from astropy.time import Time

        def summarize_through_a_leap_second(df, params, groupby_cols=None):
            Time("2015-06-30 23:59:60.354893", scale="utc").to_datetime(timezone=pytz.UTC)
            str(Time(datetime(2015, 6, 30, 23, 59, 59), scale="utc") + 1.354893 * u.s)
            return pd.DataFrame([{"night_day_ratio": 0.876}])

        monkeypatch.setattr(stats_mod.analysis, "summarize_df", summarize_through_a_leap_second)
        subject_df = pd.DataFrame({"groupby_col": [traj_gdf["groupby_col"].iloc[0]], "mature": [True]})

        result = compute_subject_stats(traj_gdf=traj_gdf, subject_df=subject_df, etd_df=etd_gdf)

        assert result.iloc[0]["night_day_ratio"] == pytest.approx(0.88)
        with pytest.raises(ValueError, match="leap second"):
            Time("2015-06-30 23:59:60.354893", scale="utc").to_datetime(timezone=pytz.UTC)


def test_night_day_ratio_param_is_built_without_stubs():
    # SummaryParam became an Annotated union in ecoscope.platform; calling it raised
    # AttributeError('__bases__'). The concrete class must construct and validate.
    param = stats_mod.NightDayRatioSummaryParam(display_name="night_day_ratio", aggregator="night_day_ratio")
    assert param.aggregator == "night_day_ratio"

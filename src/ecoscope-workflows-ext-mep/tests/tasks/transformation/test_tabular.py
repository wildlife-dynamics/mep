"""Tests for ecoscope_workflows_ext_mep.tasks.transformation._tabular.

Every function here is registered via `wt_registry.register()`, a no-op
decorator at call time, so each is called directly as plain Python
against small, hand-built DataFrames/GeoDataFrames.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from ecoscope.platform.tasks.filter._filter import TimeRange, UTC_TIMEZONEINFO
from ecoscope_workflows_ext_mep.tasks.transformation._tabular import (
    add_non_null_flag,
    add_time_since_visit,
    compute_dwell_time,
    compute_patrol_effort_fraction,
    dataframe_column_unique,
    operational_days,
    reset_dataframe_index,
)


def _time_range(since: str, until: str) -> TimeRange:
    return TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO)


class TestDataframeColumnUnique:
    def test_returns_unique_values_preserving_first_occurrence_order(self):
        df = pd.DataFrame({"subject_id": ["a", "b", "a", "c", "b"]})

        result = dataframe_column_unique(df=df, column_name="subject_id")

        assert list(result) == ["a", "b", "c"]

    def test_return_value_is_a_numpy_ndarray_not_a_list(self):
        # The return annotation says `list`, but `Series.unique()` actually
        # returns a numpy ndarray -- documenting the real runtime behavior
        # rather than the (inaccurate) type hint.
        df = pd.DataFrame({"x": [1, 2, 2, 3]})

        result = dataframe_column_unique(df=df, column_name="x")

        assert isinstance(result, np.ndarray)

    def test_empty_dataframe_returns_empty_array(self):
        df = pd.DataFrame({"x": []})

        result = dataframe_column_unique(df=df, column_name="x")

        assert len(result) == 0

    def test_missing_column_raises_key_error(self):
        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(KeyError):
            dataframe_column_unique(df=df, column_name="does_not_exist")

    def test_nan_is_included_as_a_unique_value(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 1.0, np.nan]})

        result = dataframe_column_unique(df=df, column_name="x")

        assert len(result) == 2
        assert 1.0 in result
        assert any(pd.isna(v) for v in result)


class TestResetDataframeIndex:
    def test_default_drop_false_moves_index_into_an_unnamed_index_column(self):
        df = pd.DataFrame({"x": [10, 20, 30]}, index=[5, 6, 7])

        result = reset_dataframe_index(df, drop=False)

        assert list(result.columns) == ["index", "x"]
        assert result["index"].tolist() == [5, 6, 7]
        assert result.index.tolist() == [0, 1, 2]

    def test_drop_true_discards_the_old_index_entirely(self):
        df = pd.DataFrame({"x": [10, 20, 30]}, index=[5, 6, 7])

        result = reset_dataframe_index(df, drop=True)

        assert list(result.columns) == ["x"]
        assert result.index.tolist() == [0, 1, 2]

    def test_named_index_keeps_its_name_as_the_new_column(self):
        df = pd.DataFrame({"x": [10, 20]}, index=pd.Index([5, 6], name="cell_id"))

        result = reset_dataframe_index(df, drop=False)

        assert "cell_id" in result.columns
        assert result["cell_id"].tolist() == [5, 6]


class TestAddTimeSinceVisit:
    def test_computes_hours_and_days_since_visit_from_time_range_until(self):
        df = pd.DataFrame({"last_visited": [pd.Timestamp("2024-01-01", tz="UTC")]})
        time_range = _time_range(since="2024-01-01", until="2024-01-03")

        result = add_time_since_visit(df, time_range)

        assert result["hours_since_visit"].iloc[0] == pytest.approx(48.0)
        assert result["days_since_visit"].iloc[0] == pytest.approx(2.0)

    def test_missing_last_visited_produces_nan_rather_than_raising(self):
        # An all-NaT column with no explicit tz infers as tz-naive, which
        # would fail to subtract against `time_range.until` (tz-aware) --
        # so the column must be built as tz-aware even though every value
        # is null.
        df = pd.DataFrame({"last_visited": pd.Series([pd.NaT], dtype="datetime64[ns, UTC]")})
        time_range = _time_range(since="2024-01-01", until="2024-01-03")

        result = add_time_since_visit(df, time_range)

        assert pd.isna(result["hours_since_visit"].iloc[0])
        assert pd.isna(result["days_since_visit"].iloc[0])


class TestAddNonNullFlag:
    def test_flags_non_null_rows_true_and_null_rows_false(self):
        df = pd.DataFrame({"v": [1.0, None, 3.0]})

        result = add_non_null_flag(df, source_column="v")

        assert result["is_present"].tolist() == [True, False, True]

    def test_default_flag_column_name_is_is_present(self):
        df = pd.DataFrame({"v": [1.0]})

        result = add_non_null_flag(df, source_column="v")

        assert "is_present" in result.columns

    def test_custom_flag_column_name_is_honored(self):
        df = pd.DataFrame({"v": [1.0, None]})

        result = add_non_null_flag(df, source_column="v", flag_column="visited")

        assert result["visited"].tolist() == [True, False]

    def test_missing_source_column_raises_key_error(self):
        df = pd.DataFrame({"v": [1.0]})

        with pytest.raises(KeyError):
            add_non_null_flag(df, source_column="does_not_exist")


class TestComputeDwellTime:
    def test_splits_moving_segment_time_by_length_fraction_per_cell(self):
        # Two adjacent 1x1 cells (index 0: x in [0,1], index 1: x in [1,2]).
        # One horizontal segment spans both cells evenly, so its 100s
        # timespan should split 50/50 by the fraction of length in each.
        moving = gpd.GeoDataFrame(
            {"timespan_seconds": [100], "geometry": [LineString([(0, 0.5), (2, 0.5)])]},
            crs="EPSG:3857",
        )
        trajectories = gpd.GeoDataFrame(moving, geometry="geometry", crs="EPSG:3857")
        grid = gpd.GeoDataFrame(
            {"index": [0, 1], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
            crs="EPSG:3857",
        )

        result = compute_dwell_time(patrol_trajectories=trajectories, gridded_spatial_feature=grid)

        by_cell = result.set_index("index")["seconds_in_cell"]
        assert by_cell[0] == pytest.approx(50.0)
        assert by_cell[1] == pytest.approx(50.0)

    def test_stationary_segment_contributes_full_duration_to_its_cell(self):
        # A zero-length segment (start == end) is "stationary": its whole
        # duration goes to whichever cell contains its point, not split.
        stationary = gpd.GeoDataFrame(
            {"timespan_seconds": [30], "geometry": [LineString([(0.5, 0.5), (0.5, 0.5)])]},
            crs="EPSG:3857",
        )
        trajectories = gpd.GeoDataFrame(stationary, geometry="geometry", crs="EPSG:3857")
        grid = gpd.GeoDataFrame(
            {"index": [0, 1], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
            crs="EPSG:3857",
        )

        result = compute_dwell_time(patrol_trajectories=trajectories, gridded_spatial_feature=grid)

        by_cell = result.set_index("index")["seconds_in_cell"]
        assert by_cell[0] == pytest.approx(30.0)
        assert 1 not in by_cell.index

    def test_moving_and_stationary_time_combine_in_the_same_cell(self):
        moving = gpd.GeoDataFrame(
            {"timespan_seconds": [100], "geometry": [LineString([(0, 0.5), (2, 0.5)])]},
            crs="EPSG:3857",
        )
        stationary = gpd.GeoDataFrame(
            {"timespan_seconds": [30], "geometry": [LineString([(0.5, 0.5), (0.5, 0.5)])]},
            crs="EPSG:3857",
        )
        trajectories = gpd.GeoDataFrame(
            pd.concat([moving, stationary], ignore_index=True), geometry="geometry", crs="EPSG:3857"
        )
        grid = gpd.GeoDataFrame(
            {"index": [0, 1], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
            crs="EPSG:3857",
        )

        result = compute_dwell_time(patrol_trajectories=trajectories, gridded_spatial_feature=grid)

        by_cell = result.set_index("index")["seconds_in_cell"]
        assert by_cell[0] == pytest.approx(80.0)  # 50 moving + 30 stationary
        assert by_cell[1] == pytest.approx(50.0)
        assert result.loc[result["index"] == 0, "minutes_in_cell"].iloc[0] == pytest.approx(80.0 / 60)
        assert result.loc[result["index"] == 0, "hours_in_cell"].iloc[0] == pytest.approx(80.0 / 3600)


class TestOperationalDays:
    def test_single_day_segment_counts_as_one_day(self):
        trajs = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "segment_start": ["2024-01-02 08:00"],
                "segment_end": ["2024-01-02 10:00"],
            }
        )
        time_range = _time_range(since="2024-01-01", until="2024-01-05")

        result = operational_days(trajs=trajs, groupby_cols=["subject_id"], time_range=time_range)

        assert result.loc[result["subject_id"] == "s1", "days_on_patrol"].iloc[0] == 1

    def test_segment_spanning_midnight_counts_each_calendar_day_it_touches(self):
        trajs = pd.DataFrame(
            {
                "subject_id": ["s1", "s1"],
                "segment_start": ["2024-01-02 08:00", "2024-01-03 23:00"],
                "segment_end": ["2024-01-02 10:00", "2024-01-04 01:00"],
            }
        )
        time_range = _time_range(since="2024-01-01", until="2024-01-05")

        result = operational_days(trajs=trajs, groupby_cols=["subject_id"], time_range=time_range)

        row = result.loc[result["subject_id"] == "s1"].iloc[0]
        # 01-02 (single-day) + 01-03 and 01-04 (from the midnight-spanning segment) = 3 distinct days
        assert row["days_on_patrol"] == 3

    def test_reporting_period_and_percentage_are_based_on_time_range_not_data(self):
        trajs = pd.DataFrame(
            {
                "subject_id": ["s1"],
                "segment_start": ["2024-01-02 08:00"],
                "segment_end": ["2024-01-02 10:00"],
            }
        )
        # A 5-day window (inclusive of both ends): Jan 1 through Jan 5.
        time_range = _time_range(since="2024-01-01", until="2024-01-05")

        result = operational_days(trajs=trajs, groupby_cols=["subject_id"], time_range=time_range)

        row = result.iloc[0]
        assert row["reporting_period_days"] == 5
        assert row["active_days_percentage"] == pytest.approx(1 / 5 * 100, abs=0.1)

    def test_groups_independently_by_groupby_cols(self):
        trajs = pd.DataFrame(
            {
                "subject_id": ["s1", "s2"],
                "segment_start": ["2024-01-02 08:00", "2024-01-02 08:00"],
                "segment_end": ["2024-01-02 10:00", "2024-01-03 10:00"],
            }
        )
        time_range = _time_range(since="2024-01-01", until="2024-01-05")

        result = operational_days(trajs=trajs, groupby_cols=["subject_id"], time_range=time_range)

        by_subject = result.set_index("subject_id")["days_on_patrol"]
        assert by_subject["s1"] == 1
        assert by_subject["s2"] == 2


class TestComputePatrolEffortFraction:
    def test_returns_percentage_of_area_that_is_not_unvisited(self):
        gdf = gpd.GeoDataFrame(
            {
                "visit_bin": ["Unvisited", "5-10"],
                "geometry": [box(0, 0, 10, 10), box(10, 0, 20, 10)],
            },
            crs="EPSG:3857",
        )

        result = compute_patrol_effort_fraction(gdf)

        assert result == pytest.approx(50.0)

    def test_all_unvisited_returns_zero(self):
        gdf = gpd.GeoDataFrame(
            {"visit_bin": ["Unvisited", "Unvisited"], "geometry": [box(0, 0, 10, 10), box(10, 0, 20, 10)]},
            crs="EPSG:3857",
        )

        assert compute_patrol_effort_fraction(gdf) == pytest.approx(0.0)

    def test_zero_total_area_returns_zero_without_dividing_by_zero(self):
        gdf = gpd.GeoDataFrame({"visit_bin": ["Unvisited"], "geometry": [box(0, 0, 0, 0)]}, crs="EPSG:3857")

        assert compute_patrol_effort_fraction(gdf) == 0.0

    def test_missing_visit_bin_column_raises_key_error(self):
        gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:3857")

        with pytest.raises(KeyError):
            compute_patrol_effort_fraction(gdf)

    def test_geographic_crs_raises_value_error(self):
        gdf = gpd.GeoDataFrame({"visit_bin": ["5-10"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")

        with pytest.raises(ValueError, match="geographic"):
            compute_patrol_effort_fraction(gdf)

    def test_missing_crs_raises_value_error(self):
        gdf = gpd.GeoDataFrame({"visit_bin": ["5-10"], "geometry": [box(0, 0, 1, 1)]})

        with pytest.raises(ValueError, match="no CRS"):
            compute_patrol_effort_fraction(gdf)

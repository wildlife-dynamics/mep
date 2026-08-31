"""Tests for ecoscope_workflows_ext_mep.tasks.results._sitrep.

`compile_sitrep` and `get_sitrep_event_config` are registered via
`wt_registry.register()`, a no-op decorator at call time, so they are
called directly as plain Python. The EarthRanger IO client is faked with a
small stub object (`_FakeERClient`, mirroring the convention in
`tests/tasks/io/test_earthranger.py`) rather than deep-mocked, so tests
assert on real behavior of the JSON-flattening / column-renaming pipeline
in `_download_events` (via `ecoscope.platform`'s real `normalize_column`),
not on mock call assertions.

The module's non-registered helpers (the per-event-type `sitrep_*`
formatters, `_format_timestamp`, `_clean_event_dataframes`,
`_compile_and_format_sitrep`, `_download_events`, `_download_all_events`)
are exercised directly too, since most of the interesting edge-case
behavior lives there.
"""

from __future__ import annotations

from datetime import datetime

import geopandas as gpd
import pandas as pd
import pytest
from ecoscope.platform.tasks.filter._filter import TimeRange, TimezoneInfo
from shapely.geometry import Point

from ecoscope_workflows_ext_mep.tasks.results._sitrep import (
    _clean_event_dataframes,
    _compile_and_format_sitrep,
    _download_all_events,
    _download_events,
    _format_timestamp,
    compile_sitrep,
    get_sitrep_event_config,
    sitrep_arrests,
    sitrep_hwc,
    sitrep_illegal_bushmeat,
    sitrep_illegal_charcoal,
    sitrep_illegal_logging,
    sitrep_mike,
    sitrep_wildlife_trap,
)


@pytest.fixture
def time_range() -> TimeRange:
    tz = TimezoneInfo(label="UTC", tzCode="UTC", name="UTC", utc="+00:00")
    return TimeRange(since=datetime(2024, 1, 1), until=datetime(2024, 6, 1), timezone=tz)


class _FakeERClient:
    """Returns a canned GeoDataFrame per event_type id, or raises."""

    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.calls: list[dict] = []

    def get_events(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.get(kwargs["event_type"], gpd.GeoDataFrame())
        if isinstance(response, Exception):
            raise response
        return response.copy()


# --------------------------------------------------------------------------- #
# get_sitrep_event_config                                                     #
# --------------------------------------------------------------------------- #


class TestGetSitrepEventConfig:
    def test_covers_all_six_expected_event_keys(self):
        config = get_sitrep_event_config()

        assert set(config.keys()) == {
            "MEP-Arrest",
            "MEP-HWC-Event",
            "MEP-Illegal-Logging",
            "MEP-Illegal-Charcoal",
            "MEP-Wildlife-Trap",
            "MEP-Illegal-Bushmeat",
        }

    def test_default_region_column_is_region(self):
        config = get_sitrep_event_config()

        assert all(v["region"] == "region" for v in config.values())

    def test_custom_region_column_is_used(self):
        config = get_sitrep_event_config(region_column="admin_area")

        assert all(v["region"] == "admin_area" for v in config.values())

    def test_each_entry_has_a_unique_event_id_and_callable_formatter(self):
        config = get_sitrep_event_config()

        event_ids = [v["event_id"] for v in config.values()]
        assert len(event_ids) == len(set(event_ids))
        assert all(callable(v["sitrep_func"]) for v in config.values())


# --------------------------------------------------------------------------- #
# sitrep_* formatters                                                         #
# --------------------------------------------------------------------------- #


class TestSitrepFormatters:
    def test_sitrep_illegal_charcoal(self):
        row = pd.Series(
            {
                "bag_count": 3,
                "kiln_count": 1,
                "destroyed": True,
                "tree_species": "Acacia",
                "transport_method": "Truck",
                "details": "found near river",
            }
        )

        result = sitrep_illegal_charcoal(row)

        assert result == "Bags:3,Kilns:1,Destroyed:True,Tree Species:Acacia,Transport:Truck.Details:found near river"

    def test_sitrep_illegal_logging(self):
        row = pd.Series({"loggingRecovered": 5, "logging_description": "oak", "details": "note"})

        result = sitrep_illegal_logging(row)

        assert result == "Log Count:5,Tree Type:oak,Timber:None.Details:note"

    def test_sitrep_wildlife_trap(self):
        row = pd.Series(
            {
                "illegal_wildlife_trap_type": "snare",
                "num_recovered": 2,
                "target_species": "impala",
                "details": "near waterhole",
            }
        )

        result = sitrep_wildlife_trap(row)

        assert result == "Trap Type:snare,Num Recovered:2,Species:impala.Details:near waterhole"

    def test_sitrep_mike(self):
        row = pd.Series(
            {
                "TypeOfDeath": "Poaching",
                "CauseOfDeath": "Gunshot",
                "ElephantSex": "Female",
                "CarcassAgeClass": "Adult",
                "TuskStatus": "Removed",
                "details": "found decomposed",
            }
        )

        result = sitrep_mike(row)

        assert result == (
            "Type of Death:Poaching,Cause of Death:Gunshot,Sex:Female,"
            "Carcass Age:Adult,Tusk Status:Removed.Details:found decomposed"
        )

    def test_sitrep_hwc(self):
        row = pd.Series(
            {
                "hwc_type": "crop_raid",
                "crop_type": "maize",
                "farm_size": "2 acres",
                "hwcmitigationrep_mitigation_action": "fence",
                "lt_success_index": 1,
                "details": "raided overnight",
            }
        )

        result = sitrep_hwc(row)

        assert result == (
            "HWC Type:crop_raid,Crop Type:maize,Farm Size:2 acres."
            "Mitigation Action: fence, Success Index: 1. Details: raided overnight"
        )

    def test_sitrep_illegal_bushmeat(self):
        row = pd.Series({"bushmeatRecovered": "Yes", "details": "3 carcasses"})

        result = sitrep_illegal_bushmeat(row)

        assert result == "Bushmeat:Yes, Details:3 carcasses"


class TestSitrepArrests:
    def test_all_categories_present(self):
        row = pd.Series(
            {
                "FirearmsRecovered": ["rifle", None],
                "BushmeatRecovered": [{"BushmeatKgs": 10}, {}],
                "SkinsRecovered": [{"SkinsNumber": 2}],
                "TusksRecovered": {"EleTuskKgs": 5, "EleTuskPieces": 2},
                "RhinoHornRecovered": {"RhinoHornKgs": 1, "RhinoHornPieces": 1},
                "ExhibitsRecoveredNotes": "extra notes",
            }
        )

        result = sitrep_arrests(row)

        assert "Firearms: 1" in result
        assert "Bushmeat Kgs: 1" in result
        assert "Skins: 1" in result
        assert "Tusks: 2 (5 kgs)" in result
        assert "RhinoHorn: 1 (1 kgs)" in result
        assert "Details: extra notes" in result

    def test_falls_back_to_reported_by_note_when_no_exhibit_notes(self):
        row = pd.Series({"reported_by__additional__note": "fallback note"})

        result = sitrep_arrests(row)

        assert result == "Details: fallback note"

    def test_zero_counts_are_omitted(self):
        row = pd.Series(
            {
                "FirearmsRecovered": [],
                "BushmeatRecovered": [],
                "SkinsRecovered": [],
                "TusksRecovered": 0,
                "RhinoHornRecovered": 0,
                "details": "nothing recovered",
            }
        )

        result = sitrep_arrests(row)

        assert "Firearms" not in result
        assert "Bushmeat" not in result
        assert "Skins" not in result
        assert "Tusks" not in result
        assert "RhinoHorn" not in result


# --------------------------------------------------------------------------- #
# _format_timestamp                                                           #
# --------------------------------------------------------------------------- #


class TestFormatTimestamp:
    def test_pandas_timestamp_is_isoformatted(self):
        ts = pd.Timestamp("2024-03-01T12:00:00")

        assert _format_timestamp(ts) == "2024-03-01T12:00:00"

    def test_datetime_is_isoformatted(self):
        dt = datetime(2024, 3, 1, 12, 0, 0)

        assert _format_timestamp(dt) == "2024-03-01T12:00:00"

    def test_string_passes_through_unchanged(self):
        assert _format_timestamp("2024-03-01") == "2024-03-01"

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported timestamp type"):
            _format_timestamp(12345)


# --------------------------------------------------------------------------- #
# _clean_event_dataframes                                                     #
# --------------------------------------------------------------------------- #


class TestCleanEventDataframes:
    def test_keeps_only_selected_columns_that_exist(self):
        df = pd.DataFrame({"time": ["t"], "event_type": ["x"], "junk": ["y"]})

        result = _clean_event_dataframes([df], selected_cols=["time", "event_type", "name"])

        assert list(result[0].columns) == ["time", "event_type"]

    def test_drops_frames_with_no_matching_columns(self):
        df = pd.DataFrame({"unrelated": [1]})

        result = _clean_event_dataframes([df], selected_cols=["time", "event_type"])

        assert result == []

    def test_drops_duplicate_columns_keeping_first(self):
        df = pd.DataFrame([[1, 2]], columns=["time", "time"])

        result = _clean_event_dataframes([df], selected_cols=["time"])

        assert list(result[0].columns) == ["time"]
        assert result[0]["time"].iloc[0] == 1


# --------------------------------------------------------------------------- #
# _compile_and_format_sitrep                                                  #
# --------------------------------------------------------------------------- #


class TestCompileAndFormatSitrep:
    def test_sorts_by_time_descending_and_renames_columns(self):
        events = [
            pd.DataFrame({"time": [pd.Timestamp("2024-01-01")], "event_type": ["Arrest"]}),
            pd.DataFrame({"time": [pd.Timestamp("2024-06-01")], "event_type": ["HWC Event"]}),
        ]

        result = _compile_and_format_sitrep(events, df_cols={"event_type": "type"})

        assert list(result["type"]) == ["HWC Event", "Arrest"]
        assert list(result["time"]) == ["01-Jun-2024", "01-Jan-2024"]


# --------------------------------------------------------------------------- #
# _download_events                                                            #
# --------------------------------------------------------------------------- #


class TestDownloadEvents:
    def _params(self, sitrep_func=sitrep_illegal_bushmeat, event_type="Illegal Bushmeat", region="region"):
        return {"sitrep_func": sitrep_func, "event_type": event_type, "event_id": "some-id", "region": region}

    def test_assertion_error_from_client_returns_empty_geodataframe(self):
        class RaisesAssertion:
            def get_events(self, **kwargs):
                raise AssertionError("no data")

        result = _download_events(RaisesAssertion(), self._params(), "since", "until")

        assert result.empty

    def test_missing_event_details_column_returns_empty_geodataframe(self):
        class NoDetails:
            def get_events(self, **kwargs):
                return gpd.GeoDataFrame({"serial_number": [1], "geometry": [Point(0, 0)]})

        result = _download_events(NoDetails(), self._params(), "since", "until")

        assert result.empty

    def test_flattens_event_details_and_computes_lat_lon(self):
        df = gpd.GeoDataFrame(
            {
                "serial_number": [1, 2],
                "time": ["2024-03-01T00:00:00Z", "2024-03-02T00:00:00Z"],
                "region": ["Loita", "Loita"],
                "event_details": [
                    {"bushmeatRecovered": "Yes", "details": "note a"},
                    {"bushmeatRecovered": "No", "details": "note b"},
                ],
                "geometry": [Point(35.0, -1.0), Point(35.1, -1.1)],
            },
            geometry="geometry",
        )

        class Client:
            def get_events(self, **kwargs):
                return df.copy()

        result = _download_events(Client(), self._params(), "since", "until")

        assert list(result.index) == [1, 2]
        assert result.loc[1, "bushmeatRecovered"] == "Yes"
        assert result.loc[1, "sitrep_comment"] == "Bushmeat:Yes, Details:note a"
        assert result.loc[1, "event_type"] == "Illegal Bushmeat"
        assert result.loc[1, "latitude"] == -1.0
        assert result.loc[1, "longitude"] == 35.0
        assert result["TusksRecovered"].tolist() == [0, 0]
        assert result["RhinoHornRecovered"].tolist() == [0, 0]

    def test_legacy_details_capitalization_is_renamed(self):
        df = gpd.GeoDataFrame(
            {
                "serial_number": [1],
                "time": ["2024-01-01T00:00:00Z"],
                "region": ["Loita"],
                "event_details": [{"loggingRecovered": 3, "logging_description": "oak", "Details": "legacy field"}],
                "geometry": [Point(1, 1)],
            },
            geometry="geometry",
        )

        class Client:
            def get_events(self, **kwargs):
                return df.copy()

        result = _download_events(
            Client(), self._params(sitrep_func=sitrep_illegal_logging, event_type="Illegal Logging"), "s", "u"
        )

        assert result.loc[1, "details"] == "legacy field"
        assert "Details" not in result.columns


# --------------------------------------------------------------------------- #
# _download_all_events                                                       #
# --------------------------------------------------------------------------- #


class TestDownloadAllEvents:
    def test_one_source_failing_does_not_stop_the_others(self, time_range):
        config = get_sitrep_event_config()
        ok_df = gpd.GeoDataFrame(
            {
                "serial_number": [1],
                "time": ["2024-03-01T00:00:00Z"],
                "region": ["Loita"],
                "event_details": [{"bushmeatRecovered": "Yes", "details": "x"}],
                "geometry": [Point(1, 1)],
            },
            geometry="geometry",
        )
        client = _FakeERClient(
            {
                config["MEP-Arrest"]["event_id"]: RuntimeError("network blew up"),
                config["MEP-Illegal-Bushmeat"]["event_id"]: ok_df,
            }
        )
        selected = {k: config[k] for k in ["MEP-Arrest", "MEP-Illegal-Bushmeat"]}

        result = _download_all_events(client, selected, time_range)

        assert len(result) == 1
        assert len(result[0]) == 1

    def test_sources_with_no_events_are_skipped(self, time_range):
        config = get_sitrep_event_config()
        client = _FakeERClient({})  # every call returns an empty GeoDataFrame
        selected = {k: config[k] for k in ["MEP-Arrest", "MEP-HWC-Event"]}

        result = _download_all_events(client, selected, time_range)

        assert result == []


# --------------------------------------------------------------------------- #
# compile_sitrep                                                              #
# --------------------------------------------------------------------------- #


class TestCompileSitrep:
    def test_no_events_returns_empty_dataframe(self, time_range):
        config = get_sitrep_event_config()
        client = _FakeERClient({})

        result = compile_sitrep(client, {"MEP-Arrest": config["MEP-Arrest"]}, time_range)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_events_from_multiple_sources_are_merged_sorted_and_renamed(self, time_range):
        config = get_sitrep_event_config()
        bushmeat_df = gpd.GeoDataFrame(
            {
                "serial_number": [1],
                "time": ["2024-03-01T00:00:00Z"],
                "region": ["Loita"],
                "name": ["Bushmeat A"],
                "event_details": [{"bushmeatRecovered": "Yes", "details": "note a"}],
                "geometry": [Point(35.0, -1.0)],
            },
            geometry="geometry",
        )
        hwc_df = gpd.GeoDataFrame(
            {
                "serial_number": [2],
                "time": ["2024-05-01T00:00:00Z"],
                "region": ["Mara"],
                "name": ["HWC A"],
                "event_details": [
                    {"hwc_type": "crop_raid", "crop_type": "maize", "lt_success_index": 1, "details": "note b"}
                ],
                "geometry": [Point(35.2, -1.2)],
            },
            geometry="geometry",
        )
        client = _FakeERClient(
            {
                config["MEP-Illegal-Bushmeat"]["event_id"]: bushmeat_df,
                config["MEP-HWC-Event"]["event_id"]: hwc_df,
            }
        )
        selected = {k: config[k] for k in ["MEP-Illegal-Bushmeat", "MEP-HWC-Event", "MEP-Arrest"]}

        result = compile_sitrep(client, selected, time_range)

        assert list(result["date"]) == ["01-May-2024", "01-Mar-2024"]  # sorted descending
        assert list(result["event_type"]) == ["HWC Event", "Illegal Bushmeat"]
        assert set(result.columns) == {"date", "event_type", "name", "region", "details", "latitude", "longitude"}

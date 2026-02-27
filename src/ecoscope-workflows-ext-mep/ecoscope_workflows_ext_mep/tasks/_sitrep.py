import pandas as pd
from datetime import datetime
import geopandas as gpd
from typing import Dict, Any, List
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.annotations import AnyDataFrame
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerClient
from ecoscope_workflows_ext_ecoscope.tasks.transformation._normalize import normalize_json_column

# Sitrep formatting functions
def sitrep_illegal_charcoal(x: pd.Series) -> str:
    """Format illegal charcoal event details."""
    return "Bags:{0},Kilns:{1},Destroyed:{2},Tree Species:{3},Transport:{4}.Details:{5}".format(
        x["bag_count"], x["kiln_count"], x["destroyed"], x["tree_species"], x["transport_method"], x["details"]
    )


def sitrep_illegal_logging(x: pd.Series) -> str:
    """Format illegal logging event details."""
    return "Log Count:{0},Tree Type:{1},Timber:{2}.Details:{3}".format(
        x["loggingRecovered"], x["logging_description"], x.get("bag_count"), x["details"]
    )


def sitrep_wildlife_trap(x: pd.Series) -> str:
    """Format wildlife trap event details."""
    return "Trap Type:{0},Num Recovered:{1},Species:{2}.Details:{3}".format(
        x["illegal_wildlife_trap_type"], x["num_recovered"], x["target_species"], x["details"]
    )


def sitrep_mike(x: pd.Series) -> str:
    """Format MIKE (elephant mortality) event details."""
    return "Type of Death:{0},Cause of Death:{1},Sex:{2},Carcass Age:{3},Tusk Status:{4}.Details:{5}".format(
        x["TypeOfDeath"], x["CauseOfDeath"], x["ElephantSex"], x["CarcassAgeClass"], x["TuskStatus"], x["details"]
    )


def sitrep_hwc(x: pd.Series) -> str:
    """Format Human-Wildlife Conflict event details."""
    return "HWC Type:{0},Crop Type:{1},Farm Size:{2}.Mitigation Action: {3}, Success Index: {4}. Details: {5}".format(
        x["hwc_type"],
        x["crop_type"],
        x.get("farm_size"),
        x.get("hwcmitigationrep_mitigation_action"),
        x["lt_success_index"],
        x["details"],
    )


def sitrep_illegal_bushmeat(x: pd.Series) -> str:
    """Format illegal bushmeat event details."""
    return "Bushmeat:{0}, Details:{1}".format(x["bushmeatRecovered"], x["details"])


def sitrep_arrests(x: pd.Series) -> str:
    """Format arrest event details with recovered items."""
    details_string = []

    def dict_arr(str_val: str) -> list:
        """Safely get list value from series."""
        return x[str_val] if isinstance(x.get(str_val), list) else []

    # Firearms
    firearm_count = len([f for f in dict_arr("FirearmsRecovered") if f])
    if firearm_count > 0:
        details_string.append(f"Firearms: {firearm_count}")

    # Bushmeat
    bushmeat_kgs = len([b for b in dict_arr("BushmeatRecovered") if b.get("BushmeatKgs")])
    if bushmeat_kgs > 0:
        details_string.append(f"Bushmeat Kgs: {bushmeat_kgs}")

    # Skins
    skins_count = len([s for s in dict_arr("SkinsRecovered") if s.get("SkinsNumber")])
    if skins_count > 0:
        details_string.append(f"Skins: {skins_count}")

    # Elephant tusks
    tusks = x.get("TusksRecovered", {})
    if isinstance(tusks, dict):
        tusk_kgs = tusks.get("EleTuskKgs", 0)
        tusk_pieces = tusks.get("EleTuskPieces", 0)
    else:
        tusk_kgs = tusk_pieces = 0

    if tusk_pieces > 0:
        details_string.append(f"Tusks: {tusk_pieces} ({tusk_kgs} kgs)")

    # Rhino horn
    rhinohorn = x.get("RhinoHornRecovered", {})
    if isinstance(rhinohorn, dict):
        rhinohorn_kgs = rhinohorn.get("RhinoHornKgs", 0)
        rhinohorn_pieces = rhinohorn.get("RhinoHornPieces", 0)
    else:
        rhinohorn_kgs = rhinohorn_pieces = 0

    if rhinohorn_pieces > 0:
        details_string.append(f"RhinoHorn: {rhinohorn_pieces} ({rhinohorn_kgs} kgs)")

    # Additional notes
    if x.get("ExhibitsRecoveredNotes") is not None:
        details_string.append(f"Details: {x['ExhibitsRecoveredNotes']}")
    else:
        details_string.append(f"Details: {x.get('reported_by__additional__note', '')}")

    return ", ".join(details_string)


@task
def get_sitrep_event_config(region_column: str = "region") -> Dict[str, Dict[str, Any]]:
    """
    Get configuration for sitrep event types.

    Args:
        region_column: Name of the column containing region information

    Returns:
        Dictionary mapping event keys to their configuration including
        formatting function, event type name, event ID, and region column
    """
    return {
        "MEP-Arrest": {
            "sitrep_func": sitrep_arrests,
            "event_type": "Arrest",
            "event_id": "e695fd5a-b53b-402e-946e-f9895adcd173",
            "region": region_column,
        },
        "MEP-HWC-Event": {
            "sitrep_func": sitrep_hwc,
            "event_type": "HWC Event",
            "event_id": "eacf6949-4b33-4bbf-ac70-821cf1ab1a5e",
            "region": region_column,
        },
        "MEP-Illegal-Logging": {
            "sitrep_func": sitrep_illegal_logging,
            "event_type": "Illegal Logging",
            "event_id": "f6a36871-f1b2-4e75-bac1-feeaca889f44",
            "region": region_column,
        },
        "MEP-Illegal-Charcoal": {
            "sitrep_func": sitrep_illegal_charcoal,
            "event_type": "Illegal Charcoal",
            "event_id": "403431c4-9dbd-4f44-b6e4-494ca78d9da4",
            "region": region_column,
        },
        "MEP-Wildlife-Trap": {
            "sitrep_func": sitrep_wildlife_trap,
            "event_type": "Wildlife Trap",
            "event_id": "201c02fc-db21-4323-b644-d703dd32d6a1",
            "region": region_column,
        },
        "MEP-Illegal-Bushmeat": {
            "sitrep_func": sitrep_illegal_bushmeat,
            "event_type": "Illegal Bushmeat",
            "event_id": "2b198b8a-aea0-4a82-b4d6-5c235b903489",
            "region": region_column,
        },
    }


def _download_events(er_io, params, since_filter, until_filter):
    try:
        df = er_io.get_events(
            event_type=params["event_id"],
            since=since_filter,
            until=until_filter,
            include_details=True,
            raise_on_empty=False,
            include_null_geometry=True,
            include_updates=False,
            include_related_events=False,
            include_display_values=True,
        )
    except AssertionError:
        return gpd.GeoDataFrame()
    if "event_details" not in df.columns:
        print("event_details column not found in DataFrame. Ending execution.")
        return gpd.GeoDataFrame()
    else:
        for column in ["event_details", "reported_by"]:
            if column in df.columns:
                normalize_json_column(df, column)

        df.columns = [
            col.replace("event_details__", "") if col.startswith("event_details__") else col for col in df.columns
        ]
        df.columns = df.columns.str.replace(r"^reported_by__", "", regex=True)
        df.set_index("serial_number", inplace=True)
        if "TusksRecovered" not in df.columns:
            df["TusksRecovered"] = 0
        if "RhinoHornRecovered" not in df.columns:
            df["RhinoHornRecovered"] = 0

        if "Details" in df.columns:  # HAE+ - Heuristica Aucta Est
            df.rename(columns={"Details": "details"}, inplace=True)

        df["time"] = pd.to_datetime(df["time"])
        df["sitrep_comment"] = df.apply(params["sitrep_func"], axis=1)
        df["event_type"] = params["event_type"]
        df["region"] = df[params["region"]]
        df_locations = df[~df.geometry.is_empty]
        df["latitude"] = df_locations.geometry.y
        df["longitude"] = df_locations.geometry.x
        return df


def _download_all_events(
    er_io,
    event_details: Dict[str, Any],
    time_range: TimeRange,
) -> List[AnyDataFrame]:
    """Download events from all configured sources."""
    downloaded_events = []

    # Convert TimeRange to string format for API
    since_str = _format_timestamp(time_range.since)
    until_str = _format_timestamp(time_range.until)

    print(f"Downloading events from {since_str} to {until_str}")

    for event_key, event_config in event_details.items():
        try:
            df = _download_events(
                er_io,
                event_config,
                since_str,
                until_str,
            )

            if not df.empty:
                downloaded_events.append(df)
                print(f"Downloaded {len(df)} events for {event_key}")
            else:
                print(f"No events found for {event_key}")

        except Exception as e:
            print(f"Failed to download events for {event_key}: {e}", exc_info=True)
            continue

    return downloaded_events


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    """
    Convert pandas Timestamp to ISO format string for API calls.

    Args:
        timestamp: pandas Timestamp object or datetime object

    Returns:
        ISO formatted datetime string
    """
    if isinstance(timestamp, (pd.Timestamp, datetime)):
        return timestamp.isoformat()
    elif isinstance(timestamp, str):
        return timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")


def _clean_event_dataframes(
    dataframes: List[AnyDataFrame],
    selected_cols: List[str],
) -> List[AnyDataFrame]:
    """Clean and standardize event dataframes."""
    cleaned_events = []

    for df in dataframes:
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        cols_to_keep = [col for col in selected_cols if col in df.columns]

        if not cols_to_keep:
            print("No valid columns found in dataframe. Skipping.")
            continue

        df = df[cols_to_keep]
        cleaned_events.append(df)

    return cleaned_events


def _compile_and_format_sitrep(
    cleaned_events: List[AnyDataFrame],
    df_cols: Dict[str, str],
) -> AnyDataFrame:
    """Compile events and apply final formatting."""
    sitrep_df = pd.concat(cleaned_events, axis=0, ignore_index=True)
    sitrep_df.sort_values("time", inplace=True, ascending=False)
    sitrep_df["time"] = sitrep_df["time"].dt.strftime("%d-%b-%Y")
    sitrep_df.rename(columns=df_cols, inplace=True)
    print(f"Compiled sitrep with {len(sitrep_df)} total events")
    return sitrep_df


@task
def compile_sitrep(
    er_io: EarthRangerClient,
    event_details: Dict[str, Any],
    time_range: TimeRange,
) -> AnyDataFrame:
    """
    Compile situation report from multiple event sources.

    Args:
        er_io: Earth Ranger IO client
        event_details: Dictionary of event configurations
        time_range: TimeRange object specifying start and end times
        df_cols: Dictionary mapping original column names to display names
        selected_cols: List of columns to retain (uses defaults if None)

    Returns:
        DataFrame with compiled and formatted events, sorted by date descending
    """
    selected_cols = ["time", "event_type", "name", "region", "sitrep_comment", "latitude", "longitude"]
    df_cols = {
        "time": "date",
        "event_type": "event_type",
        "name": "name",
        "region": "region",
        "sitrep_comment": "details",
    }
    downloaded_events = _download_all_events(er_io, event_details, time_range)

    if not downloaded_events:
        print("No events to compile. Returning empty DataFrame.")
        return pd.DataFrame()
    cleaned_events = _clean_event_dataframes(downloaded_events, selected_cols)
    sitrep_df = _compile_and_format_sitrep(cleaned_events, df_cols)

    return sitrep_df

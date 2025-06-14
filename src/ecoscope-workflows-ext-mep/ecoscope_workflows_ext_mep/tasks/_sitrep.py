# import getpass
# import json
# import os
# import sys

# import docx
# import fsspec
# import geopandas as gpd
# import pandas as pd
# from dotenv import load_dotenv
# from ecoscope_workflows_ext_ecoscope.connections import EarthRangerConnection
# from ecoscope_workflows_ext_ecoscope.tasks.transformation import normalize_column
# from pandas.tseries.offsets import DateOffset
# from sitrep_utils import (
#     sitrep_arrests,
#     sitrep_hwc,
#     sitrep_illegal_bushmeat,
#     sitrep_illegal_charcoal,
#     sitrep_illegal_logging,
#     sitrep_wildlife_trap,
# )

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import helper

# load_dotenv()

# # Setup logger
# logger = helper.logger


# def sitrep_illegal_charcoal(x):
#     return "Bags: {0}, Kilns: {1}, Destroyed: {2}, Tree Species: {3}, Transport: {4}. Details: {5}".format(
#         x["bag_count"], x["kiln_count"], x["destroyed"], x["tree_species"], x["transport_method"], x["details"]
#     )


# def sitrep_illegal_logging(x):
#     return "Log Count: {0}, Tree Type: {1}, Timber: {2}. Details: {3}".format(
#         x["loggingRecovered"], x["logging_description"], x.get("bag_count"), x["details"]
#     )


# def sitrep_wildlife_trap(x):
#     return "Trap Type: {0}, Num Recovered: {1}, Species: {2}. Details: {3}".format(
#         x["illegal_wildlife_trap_type"], x["num_recovered"], x["target_species"], x["details"]
#     )


# def sitrep_mike(x):
#     return """
# Type of Death: {0}, Cause of Death: {1}, Sex: {2}, Carcass Age: {3}, Tusk Status: {4}. Details: {5}""".format(
#         x["TypeOfDeath"], x["CauseOfDeath"], x["ElephantSex"], x["CarcassAgeClass"], x["TuskStatus"], x["details"]
#     )


# def sitrep_hwc(x):
#     return """
# HWC Type: {0}, Crop Type: {1}, Farm Size: {2}. Mitigation Action: {3}, Success Index: {4}. Details: {5}""".format(
#         x["hwc_type"],
#         x["crop_type"],
#         x.get("farm_size"),
#         x.get("hwcmitigationrep_mitigation_action"),
#         x["lt_success_index"],
#         x["details"],
#     )


# def sitrep_illegal_bushmeat(x):
#     return "Bushmeat : {0}, Details: {1}".format(x["bushmeatRecovered"], x["details"])


# def sitrep_arrests(x):
#     details_string = []

#     def dict_arr(str_val):
#         return x[str_val] if isinstance(x.get(str_val), list) else []

#     firearm_count = len([f for f in dict_arr("FirearmsRecovered") if f])
#     if firearm_count > 0:
#         details_string.append("Firearms: {0}".format(firearm_count))

#     bushmeat_kgs = len([b for b in dict_arr("BushmeatRecovered") if b.get("BushmeatKgs")])
#     if bushmeat_kgs > 0:
#         details_string.append("Bushmeat Kgs: {0}".format(bushmeat_kgs))

#     skins_count = len([s for s in dict_arr("SkinsRecovered") if s.get("SkinsNumber")])
#     if skins_count > 0:
#         details_string.append("Skins: {0}".format(skins_count))

#     tusks = x.get("TusksRecovered", {})
#     if isinstance(tusks, dict):
#         tusk_kgs = tusks.get("EleTuskKgs", 0)
#         tusk_pieces = tusks.get("EleTuskPieces", 0)
#     else:
#         tusk_kgs = tusk_pieces = 0

#     if tusk_pieces > 0:
#         details_string.append("Tusks: {0} ({1} kgs)".format(tusk_pieces, tusk_kgs))

#     # Rhino Horn
#     rhinohorn = x.get("RhinoHornRecovered", {})
#     if isinstance(rhinohorn, dict):
#         rhinohorn_kgs = rhinohorn.get("RhinoHornKgs", 0)
#         rhinohorn_pieces = rhinohorn.get("RhinoHornPieces", 0)
#     else:
#         rhinohorn_kgs = rhinohorn_pieces = 0

#     if rhinohorn_pieces > 0:
#         details_string.append("RhinoHorn: {0} ({1} kgs)".format(rhinohorn_pieces, rhinohorn_kgs))

#     if x.get("ExhibitsRecoveredNotes") is not None:
#         details_string.append("Details: {0}".format(x["ExhibitsRecoveredNotes"]))
#     else:
#         details_string.append("Details: {0}".format(x.get("reported_by__additional__note", "")))

#     return ", ".join(details_string)

# def get_value(item, keyname):
#     return item.get(keyname, "")


# def _download_events(er_io, params, since_filter, until_filter):
#     print(params["event_type"])
#     try:
#         df = er_io.get_events(event_type=params["event_id"], since=since_filter, until=until_filter)
#     except AssertionError:
#         return gpd.GeoDataFrame()
#     if "event_details" not in df.columns:
#         logger.error("event_details column not found in DataFrame. Ending execution.")
#         return gpd.GeoDataFrame()
#     else:
#         for column in ["event_details", "reported_by"]:
#             if column in df.columns:
#                 normalize_column(df, column)

#         df.columns = [
#             col.replace("event_details__", "") if col.startswith("event_details__") else col for col in df.columns
#         ]
#         df.columns = df.columns.str.replace(r"^reported_by__", "", regex=True)
#         df.set_index("serial_number", inplace=True)
#         if "TusksRecovered" not in df.columns:
#             df["TusksRecovered"] = 0
#         if "RhinoHornRecovered" not in df.columns:
#             df["RhinoHornRecovered"] = 0

#         if "Details" in df.columns:  # HAE+ - Heuristica Aucta Est
#             df.rename(columns={"Details": "details"}, inplace=True)

#         df["time"] = pd.to_datetime(df["time"])
#         df["sitrep_comment"] = df.apply(params["sitrep_func"], axis=1)
#         df["event_type"] = params["event_type"]
#         print("df", df.columns)
#         df["region"] = df[params["region"]]
#         df_locations = df[~df.geometry.is_empty]  # select dataframes with no-missing location.
#         df["latitude"] = df_locations.geometry.y
#         df["longitude"] = df_locations.geometry.x
#         return df


# def compile_sitrep(er_io, event_details, config):
#     downloaded_events = []

#     for _, x in event_details.items():
#         df = _download_events(er_io, x, config["since_filter"], config["until_filter"])
#         if not df.empty:
#             downloaded_events.append(df)

#     if not downloaded_events:
#         logger.warning("No events to compile. Returning empty DataFrame.")
#         return pd.DataFrame()

#     # Keep only these columns and remove duplicated column names (keep first occurrence)
#     selected_cols = ["time", "event_type", "name", "region", "sitrep_comment", "latitude", "longitude"]
#     cleaned_events = []

#     for df in downloaded_events:
#         # Drop duplicated columns (keep first occurrence)
#         df = df.loc[:, ~df.columns.duplicated(keep="first")]

#         # Retain only the relevant selected columns that exist
#         cols_to_keep = [col for col in selected_cols if col in df.columns]
#         df = df[cols_to_keep]

#         cleaned_events.append(df)

#     s = pd.concat(cleaned_events, axis=0)

#     s.sort_values("time", inplace=True, ascending=False)
#     s["time"] = s["time"].dt.strftime("%d-%b-%Y")
#     df_cols = {
#       "time":"Date",
#       "event_type":"Event Type",
#       "reported_by__name" : "Team",
#       "region":"Region",
#       "sitrep_comment":"Details",
#       "latitude":"lat",
#       "longitude":"lon"
#    }
#     s.rename(columns=df_cols, inplace=True)
#     s = s[["Date", "Event Type", "name", "Region", "Details"]]
#     return s


# @task
# def sitrep(events):

#     event_details = {
#         "MEP-Arrest": {
#             "sitrep_func": sitrep_arrests,
#             "event_type": "Arrest",
#             "event_id": "MEP-Arrest",
#             "region": "region",
#         },
#         "MEP-HWC-Event": {
#             "sitrep_func": sitrep_hwc,
#             "event_type": "HWC Event",
#             "event_id": "MEP-HWC-Event",
#             "region": "region",
#         },
#         "MEP-Illegal-Logging": {
#             "sitrep_func": sitrep_illegal_logging,
#             "event_type": "Illegal Logging",
#             "event_id": "MEP-Illegal-Logging",
#             "region": "region",
#         },
#         "MEP-Illegal-Charcoal": {
#             "sitrep_func": sitrep_illegal_charcoal,
#             "event_type": "Illegal Charcoal",
#             "event_id": "MEP-Illegal-Charcoal",
#             "region": "region",
#         },
#         "MEP-Wildlife-Trap": {
#             "sitrep_func": sitrep_wildlife_trap,
#             "event_type": "Wildlife Trap",
#             "event_id": "MEP-Wildlife-Trap",
#             "region": "region",
#         },
#         "MEP-Illegal-Bushmeat": {
#             "sitrep_func": sitrep_illegal_bushmeat,
#             "event_type": "Illegal Bushmeat",
#             "event_id": "MEP-Illegal-Bushmeat",
#             "region": "region",
#         },
#     }

#     sitrep_df = compile_sitrep(events, event_details)
#     return sitrep_df
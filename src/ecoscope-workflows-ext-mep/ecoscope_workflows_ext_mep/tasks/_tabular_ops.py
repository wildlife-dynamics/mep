import pandas as pd 
import geopandas as gpd
from typing import Dict, Any
from shapely.geometry import Point
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame,AnyGeoDataFrame

@task
def custom_map_column(
    df:AnyDataFrame, 
    column:str, 
    mapping:Dict[str, Any] ,
    inplace_col=True
    )->AnyDataFrame:
    """
    Map values in a DataFrame column using a dictionary, with normalization
    (strip whitespace + lowercase) applied to both the column values and mapping keys.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the column to map._
    mapping : dict
        Dictionary of {original_value: mapped_value}. Keys are normalized
        (stripped + lowercased) for matching.
    default : any, optional
        Value to use for entries not found in the mapping. If None, unmapped
        values become NaN.
    inplace_col : bool, default True
        If True, overwrite the original column. If False, create a new column
        named f"{column}_mapped".

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the mapped column.
    """
    df = df.copy()

    # Normalize mapping keys
    normalized_mapping = {
        str(k).strip().lower(): v for k, v in mapping.items()
    }

    # Normalize column values and apply mapping
    normalized_col = df[column].astype(str).str.strip().str.lower()
    mapped = normalized_col.map(normalized_mapping)

    target_col = column if inplace_col else f"{column}_mapped"
    df[target_col] = mapped

    return df

@task
def herder_effectiveness(df: AnyDataFrame)->AnyDataFrame:
    """
    Cross-tabulate incident counts and livestock lost by location
    (boma vs bush) and whether a herder was present.
 
    Answers: "Does having a herder present actually reduce losses?"
    """
    livestock = df[df["is_livestock"] == True].copy()
 
    # Normalize the herder-present column to a clean boolean-ish label
    livestock["herder_present"] = (
        livestock["livestock_not_lost_in_bush_herder_present"]
        .astype(str).str.strip().str.lower()
        .map({"yes": "Herder present",
              "no": "No herder",
              "unknown": "Unknown"})
        .fillna("Unknown")
    )
 
    summary = (
        livestock
        .groupby(["boma_or_bush", "herder_present"])
        .agg(incidents=("total_killed", "size"),
             total_livestock_killed=("total_killed", "sum"),
             avg_killed_per_incident=("total_killed", "mean"))
        .round(2)
        .reset_index()
        .sort_values(["boma_or_bush", "total_livestock_killed"],
                     ascending=[True, False])
    )
    return summary

@task 
def species_by_ranch(
    df: AnyDataFrame, 
    top_n=None, 
    livestock_only=False, 
    include_share=True
    )->AnyDataFrame:
    """
    Which species are most targeted at each ranch.
 
    Parameters
    ----------
    df : DataFrame
        Incident-level data.
    top_n : int, optional
        If set, return only the top N species per ranch (by total killed).
    livestock_only : bool, default False
        If True, restrict to livestock incidents only.
    include_share : bool, default True
        Add a column with each species' share of that ranch's total kills.
 
    Returns
    -------
    Long-format DataFrame: ranch, species_killed, incidents, total_killed, share_%
    """
    data = df.copy()
    if livestock_only:
        data = data[data["is_livestock"] == True]
 
    # Drop rows where species or ranch is missing/unknown
    data = data[
        data["species_killed"].notna()
        & data["ranch"].notna()
    ]
 
    summary = (
        data
        .groupby(["ranch", "species_killed"])
        .agg(incidents=("total_killed", "size"),
             total_killed=("total_killed", "sum"))
        .reset_index()
    )
 
    if include_share:
        ranch_totals = summary.groupby("ranch")["total_killed"].transform("sum")
        summary["share_%"] = (100 * summary["total_killed"] / ranch_totals).round(1)
 
    summary = summary.sort_values(
        ["ranch", "total_killed"], ascending=[True, False]
    )
 
    if top_n is not None:
        summary = (
            summary.groupby("ranch", group_keys=False)
                   .head(top_n)
                   .reset_index(drop=True)
        )
 
    return summary.reset_index(drop=True)
   
@task 
def species_by_ranch_matrix(
    df: AnyDataFrame, 
    livestock_only=False, 
    normalize=False
    )->AnyDataFrame:
    """
    Wide-format pivot: ranches as rows, species as columns, kill counts as values.
    Handy for heatmaps or quick visual comparison.

    normalize : bool
        If True, values are row-normalized (% of ranch's total kills).
    """
    long = species_by_ranch(df, livestock_only=livestock_only, include_share=False)

    matrix = (
        long.pivot_table(index="ranch",
                         columns="species_killed",
                         values="total_killed",
                         fill_value=0)
    )

    if normalize:
        matrix = matrix.div(matrix.sum(axis=1), axis=0).mul(100).round(1)

    # Order columns by overall frequency (most-killed species first)
    col_order = matrix.sum().sort_values(ascending=False).index
    matrix = matrix[col_order]

    # Flatten: ranch becomes a regular column, no leftover index name
    matrix = matrix.reset_index()
    matrix.columns.name = None

    return matrix

@task
def utm_to_4326(
    df: AnyDataFrame, 
    x_col=str, 
    y_col=str, 
    utm_epsg=32737
    )->AnyGeoDataFrame:
    """
    Build geometry from UTM x/y columns and reproject to EPSG:4326.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with UTM easting/northing columns.
    x_col, y_col : str
        Column names for easting (X) and northing (Y).
    utm_epsg : int
        EPSG code of the source UTM zone.
        Northern hemisphere: 326XX (e.g. 32636 = zone 36N)
        Southern hemisphere: 327XX (e.g. 32737 = zone 37S)

    Returns
    -------
    gpd.GeoDataFrame in EPSG:4326.
    """
    # Ensure numeric (UTM cols sometimes come in as strings)
    x = pd.to_numeric(df[x_col], errors='coerce')
    y = pd.to_numeric(df[y_col], errors='coerce')

    # Build geometry, leaving NaN coords as missing
    geometry = [Point(xi, yi) if pd.notna(xi) and pd.notna(yi) else None
                for xi, yi in zip(x, y)]

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=f"EPSG:{utm_epsg}")
    return gdf.to_crs("EPSG:4326")
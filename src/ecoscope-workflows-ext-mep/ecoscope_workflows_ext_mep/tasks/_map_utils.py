import os
import math
import ecoscope
import traceback
from enum import Enum
import geopandas as gpd
from ecoscope_workflows_core.decorators import task
from pydantic import BaseModel, Field, field_validator
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import ViewState
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import LayerDefinition
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PointLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import LegendDefinition
from typing import Union, Dict, Optional, Literal, List, Annotated, TypedDict, Tuple
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PolygonLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_point_layer
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PolylineLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_polygon_layer
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_polyline_layer
from ecoscope_workflows_ext_ecoscope.tasks.analysis._time_density import CustomGridCellSize
from ecoscope_workflows_ext_ecoscope.tasks.analysis._create_meshgrid import create_meshgrid
from ecoscope_workflows_ext_ecoscope.tasks.analysis._calculate_feature_density import calculate_feature_density


class MapStyleConfig(BaseModel):
    styles: Dict[str, Dict] = Field(default_factory=dict)
    legend: Dict[str, List[str]] = Field(default_factory=dict)


class SupportedFormat(str, Enum):
    GPKG = ".gpkg"
    GEOJSON = ".geojson"
    SHP = ".shp"


SUPPORTED_FORMATS = [f.value for f in SupportedFormat]


class MapProcessingConfig(BaseModel):
    path: str = Field(..., description="Directory path to load geospatial files from")
    target_crs: Union[int, str] = Field(default=4326, description="Target CRS to convert maps to")
    recursive: bool = Field(default=False, description="Whether to walk folders recursively")

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Invalid path: {v}")
        return v


@task
def clean_geodataframe(
    gdf: Annotated[AnyGeoDataFrame, Field(description="The geodataframe to visualize.", exclude=True)],
) -> AnyGeoDataFrame:
    return gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]


class GeometrySummary(TypedDict):
    primary_type: Literal["Polygon", "Point", "LineString", "Other", "Mixed", "Line"]


@task
def check_shapefile_geometry_type(data: AnyGeoDataFrame) -> str:
    unique_geom_types = data.geometry.geom_type.unique()

    if len(unique_geom_types) == 1:
        geom_type = unique_geom_types[0]
        if "Polygon" in geom_type:
            primary_type = "Polygon"
        elif "Point" in geom_type:
            primary_type = "Point"
        elif "LineString" in geom_type:
            primary_type = "LineString"
        else:
            primary_type = "Other"
    else:
        primary_type = "Mixed"

    return primary_type


@task
def load_map_files(config: MapProcessingConfig) -> Dict[str, AnyGeoDataFrame]:
    """
    Loads geospatial files from the specified path and returns a dictionary
    mapping filenames to cleaned GeoDataFrames, reprojected to target CRS if needed.
    """
    path = config.path
    target_crs = config.target_crs
    recursive = config.recursive
    loaded_files: Dict[str, AnyGeoDataFrame] = {}

    walk = os.walk(path) if recursive else [(path, None, os.listdir(path))]
    for root, _, files in walk:
        for file in files:
            if not file.lower().endswith(tuple(SUPPORTED_FORMATS)):
                continue

            try:
                file_path = os.path.join(root, file)
                gdf = gpd.read_file(file_path)

                if gdf.empty:
                    print(f"Skipped empty file: {file}")
                    continue

                if gdf.crs and gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)

                loaded_files[file] = clean_geodataframe(gdf)

            except Exception as e:
                print(f"Error processing {file}: {e}")
                traceback.print_exc()

    return loaded_files


def clean_file_keys(file_dict: dict) -> dict:
    def clean_key(key: str) -> str:
        for ext in SUPPORTED_FORMATS:
            if key.lower().endswith(ext):
                key = key[: -len(ext)]
                break
        return key.replace(" and ", "_").replace(" ", "_").replace(".", "")

    return {clean_key(k): v for k, v in file_dict.items()}


@task
def create_layer_from_gdf(
    filename: str,
    gdf: AnyGeoDataFrame,
    style_config: MapStyleConfig,
    primary_type: str,
) -> Optional[object]:
    if filename not in style_config.styles:
        print(f"No style config for '{filename}'")
        return None

    style_params = style_config.styles[filename]
    legend = None

    if style_config.legend and "labels" in style_config.legend and "colors" in style_config.legend:
        legend = LegendDefinition(labels=style_config.legend["labels"], colors=style_config.legend["colors"])

    try:
        if primary_type == "Polygon":
            print(f"Creating polygon layer for '{filename}'")
            return create_polygon_layer(gdf, layer_style=PolygonLayerStyle(**style_params), legend=legend)
        elif primary_type == "Point":
            print(f"Creating point layer for '{filename}'")
            return create_point_layer(gdf, layer_style=PointLayerStyle(**style_params), legend=legend)
        elif primary_type in ("Line", "LineString"):
            print(f"Creating line layer for '{filename}'")
            return create_polyline_layer(gdf, layer_style=PolylineLayerStyle(**style_params), legend=legend)
        else:
            print(f"Unsupported geometry type '{primary_type}' for file '{filename}'")
    except Exception as e:
        print(f"Error creating layer for '{filename}': {e}")
        traceback.print_exc()

    return None


@task
def create_map_layers(file_dict: Dict[str, AnyGeoDataFrame], style_config: MapStyleConfig) -> List[LayerDefinition]:
    """
    Create styled map layers from a dictionary of GeoDataFrames using the provided style config.

    Args:
        file_dict: Dictionary mapping filenames to AnyGeoDataFrames.
        style_config: Object holding style definitions and legend config.
    Returns:
        A list of styled map layer objects.
    """
    layers: List[LayerDefinition] = []
    cleaned_files = clean_file_keys(file_dict)

    for filename, gdf in cleaned_files.items():
        try:
            geom_analysis = check_shapefile_geometry_type(gdf)
            print(f"{filename} geometry type: {geom_analysis}")
            primary_type = geom_analysis
            layer = create_layer_from_gdf(filename, gdf, style_config, primary_type)

            if layer is not None:
                layers.append(layer)
        except Exception as e:
            print(f"Error processing layer for '{filename}': {e}")
            traceback.print_exc()

    print(f"Successfully created {len(layers)} map layers")
    return layers


def _zoom_from_bbox(
    minx,
    miny,
    maxx,
    maxy,
    viewport_width_px=1000,
    viewport_height_px=1000,
    padding_frac=0.10,
    min_zoom=0.0,
    max_zoom=20.0,
    tile_size=256.0,
):
    padding_frac = max(0.0, min(0.3, float(padding_frac)))

    usable_width = max(1, int(viewport_width_px * (1.0 - 2.0 * padding_frac)))
    usable_height = max(1, int(viewport_height_px * (1.0 - 2.0 * padding_frac)))

    lat_span = max(1e-12, maxy - miny)
    lon_span = max(1e-12, maxx - minx)

    zoom_for_width = math.log2(usable_width * 360.0 / (lon_span * tile_size))
    zoom_for_height = math.log2(usable_height * 180.0 / (lat_span * tile_size))
    zoom_level = min(zoom_for_width, zoom_for_height)

    return round(max(min_zoom, min(max_zoom, zoom_level)), 2)


@task
def create_view_state_from_gdf(gdf: AnyGeoDataFrame, pitch: int = 0, bearing: int = 0) -> ViewState:
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")

    # Ensure CRS is geographic
    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0

    zoom = _zoom_from_bbox(
        minx,
        miny,
        maxx,
        maxy,
        viewport_width_px=1000,  # tweak if your map container differs
        viewport_height_px=700,
        padding_frac=0.12,  # increase to zoom out slightly; decrease to zoom in
        tile_size=512,
        min_zoom=2.0,
        max_zoom=20.0,
    )

    print(f"View State zoom: {zoom}")
    return ViewState(longitude=center_lon, latitude=center_lat, zoom=zoom, pitch=pitch, bearing=bearing)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 characters long.")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


@task
def generate_density_grid(
    features_gdf: AnyGeoDataFrame, cell_size_meters: int = 2000, geometry_type: Literal["point", "line"] = "point"
) -> AnyGeoDataFrame:
    """
    Generates a density grid based on point or line features within a defined area of interest.

    Args:
        features_gdf (AnyGeoDataFrame): The input GeoDataFrame containing features (points or lines).
        cell_size_meters (int): The size of each grid cell in meters. Default is 2000.
        geometry_type (Literal["point", "line"]): The geometry type of features to calculate density for.

    Returns:
        AnyGeoDataFrame: A filtered density grid (only cells with density > 0 and not NaN).
    """
    meshgrid_gdf = create_meshgrid(
        aoi=features_gdf, auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=cell_size_meters)
    )

    density_grid = calculate_feature_density(
        geodataframe=features_gdf, meshgrid=meshgrid_gdf, geometry_type=geometry_type
    )

    density_grid = density_grid[(density_grid["density"].notna()) & (density_grid["density"] > 0)]
    return density_grid


@task
def build_landdx_style_config(aoi_list: List[str], color_map: Dict[str, Tuple[int, int, int]]) -> MapStyleConfig:
    """
    Build a MapStyleConfig object with styles and legends for specified AOI types.

    Args:
        aoi_list (List[str]): List of area of interest (AOI) types.
        color_map (Dict[str, Tuple[int, int, int]]): Mapping from AOI type to RGB color.

    Returns:
        MapStyleConfig: A style configuration with defined styles and legends.
    """
    styles: Dict[str, Dict[str, object]] = {}
    legend: Dict[str, List[str]] = {"labels": [], "colors": []}

    for aoi in aoi_list:
        rgb = color_map.get(aoi, (128, 128, 128))  # fallback to gray
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

        styles[aoi] = {
            "get_elevation": 1000,
            "get_fill_color": rgb,
            "opacity": 0.15,
            "get_line_width": 4.00,
            "get_width": 5,
            "get_line_color": rgb,
        }

        legend["labels"].append(aoi)
        legend["colors"].append(hex_color)

    return MapStyleConfig(styles=styles, legend=legend)


@task
def download_land_dx(
    url: Annotated[str, Field(description="URL to retrieve the LandDx database")],
    path: Annotated[str, Field(description="Local path to save the LandDx database copy")],
    overwrite_existing: Annotated[bool, Field(default=False, description="Overwrite the existing file if it exists")],
    unzip: Annotated[bool, Field(default=True, description="Whether to unzip the downloaded file")],
) -> str:
    """
    Downloads the LandDx database from a given URL and saves it to the specified path.

    Parameters:
        url (str): The download URL.
        path (str): Destination path for saving the file.
        overwrite_existing (bool): Whether to overwrite an existing file. Default is False.
        unzip (bool): Whether to unzip the downloaded file. Default is True.
    """
    ecoscope.io.utils.download_file(url=url, path=path, overwrite_existing=overwrite_existing, unzip=unzip)
    return path


@task
def load_landdx_aoi(map_path: str, aoi: List[str]) -> Optional[AnyGeoDataFrame]:
    """
    Recursively search for 'landDx.gpkg' in the given path, load it, and filter by area types (AOI).

    Args:
        map_path (str): Directory path to search recursively for the GeoPackage.
        aoi (List[str]): Area of interest types to filter for (e.g., ["Community Conservancy"]).

    Returns:
        Optional[gpd.GeoDataFrame]: Filtered GeoDataFrame, or None if not found or fails to load.
    """
    landDx_path = None

    # Search recursively
    for root, _, files in os.walk(map_path):
        if "landDx.gpkg" in files:
            landDx_path = os.path.join(root, "landDx.gpkg")
            break

    if landDx_path is None:
        print("landDx.gpkg not found in the specified path.")
        return None

    # Try to load and filter
    try:
        geodataframe = gpd.read_file(landDx_path, layer="landDx_polygons").set_index("globalid")
        filtered = geodataframe[geodataframe["type"].isin(aoi)]
        print(f"Loaded landDx.gpkg — total: {len(geodataframe)}, filtered: {len(filtered)}")
        return filtered

    except Exception as e:
        print(f"Error loading or filtering landDx.gpkg: {e}")
        return None


@task
def annotate_gdf_dict_with_geometry_type(gdf_dict: Dict[str, AnyGeoDataFrame]) -> Dict[str, Dict[str, object]]:
    """
    Annotates each GeoDataFrame in the dictionary with its primary geometry type.

    Args:
        gdf_dict (Dict[str, gpd.GeoDataFrame]): Dictionary of GeoDataFrames.

    Returns:
        Dict[str, Dict]: Dictionary with keys preserved, and each value being a dict
                         containing the original GeoDataFrame and its geometry type.
    """
    result = {}

    # others to be added soon
    for name, gdf in gdf_dict.items():
        unique_geom_types = gdf.geometry.geom_type.unique()

        if len(unique_geom_types) == 1:
            geom_type = unique_geom_types[0]
            if "Polygon" in geom_type:
                primary_type = "Polygon"
            elif "Point" in geom_type:
                primary_type = "Point"
            elif "LineString" in geom_type:
                primary_type = "LineString"
            else:
                primary_type = "Other"
        else:
            primary_type = "Mixed"

        result[name] = {"gdf": gdf, "geometry_type": primary_type}

    return result


@task
def create_map_layers_from_annotated_dict(
    annotated_dict: Dict[str, Dict[str, object]], style_config: MapStyleConfig
) -> List[LayerDefinition]:
    """
    Create styled map layers from a dictionary of {name: {gdf, geometry_type}}.

    Args:
        annotated_dict: A dictionary where each value contains a GeoDataFrame and its geometry type.
        style_config: A MapStyleConfig object that defines styles for each name key.

    Returns:
        List of LayerDefinition objects for mapping.
    """
    # Convert dict to MapStyleConfig if needed
    if isinstance(style_config, dict):
        style_config = MapStyleConfig(**style_config)

    layers: List[LayerDefinition] = []

    for name, content in annotated_dict.items():
        print(f"Name: {name}")

        gdf = content.get("gdf")
        geometry_type = content.get("geometry_type")

        try:
            layer = create_layer_from_gdf(filename=name, gdf=gdf, style_config=style_config, primary_type=geometry_type)

            if layer:
                layers.append(layer)

        except Exception as e:
            print(f"Error creating layer for '{name}': {e}")
            traceback.print_exc()

    print(f"Created {len(layers)} layers from annotated dict")
    return layers


@task
def combine_map_layers(
    static_layers: Annotated[
        Union[LayerDefinition, list[LayerDefinition]], Field(description="Static layers from local files or base maps.")
    ] = [],
    grouped_layers: Annotated[
        Union[LayerDefinition, list[LayerDefinition]],
        Field(description="Grouped layers generated from split/grouped data."),
    ] = [],
) -> list[LayerDefinition]:
    """
    Combine static and grouped map layers into a single list for rendering in `draw_ecomap`.
    """
    if not isinstance(static_layers, list):
        static_layers = [static_layers]
    if not isinstance(grouped_layers, list):
        grouped_layers = [grouped_layers]

    return static_layers + grouped_layers

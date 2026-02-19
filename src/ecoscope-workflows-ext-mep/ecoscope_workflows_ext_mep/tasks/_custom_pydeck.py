import logging
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_custom.tasks.results._map import ViewState
from ecoscope_workflows_ext_ste.tasks._mapdeck_utils import _zoom_from_bbox

logger = logging.getLogger(__name__)


@task
def custom_view_state_deck_gdf(
    gdf, pitch: int = 0, bearing: int = 0, map_width_px: int = 1280, map_height_px: int = 720, buffer: float = 0.25
) -> ViewState:
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")

    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0

    cust_map_width = map_width_px - (map_width_px * buffer)
    cust_map_height = map_height_px - (map_height_px * buffer)

    zoom = _zoom_from_bbox(minx, miny, maxx, maxy, map_width_px=cust_map_width, map_height_px=cust_map_height)
    return ViewState(longitude=center_lon, latitude=center_lat, zoom=zoom, pitch=pitch, bearing=bearing)

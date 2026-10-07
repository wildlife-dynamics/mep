"""Cloud-free satellite composites from Google Earth Engine.

A Python port of the GEE-PICX approach (https://github.com/EcoDynIZW/GEE-PICX):
filter Landsat / Sentinel-2 scenes over an ROI, mask clouds, reduce the time range
to a single composite, optionally add spectral indices, and add a `valid_count`
band holding the number of clear observations per pixel.

Unlike GEE-PICX (which exports asynchronously to Google Drive), composites are
downloaded synchronously in tiles via `ee.data.computePixels` and written to a
GeoTIFF under `root_path`, so downstream tasks can use them directly.

Raster encoding:
    - reflectance and index bands are multiplied by `SCALE_FACTOR` and stored as int16
    - `valid_count` is stored unscaled: 0 = no clear observation inside the ROI
    - pixels outside the ROI are `NODATA` in every band
"""

import hashlib
import math
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope.platform.connections import EarthEngineClient
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

NODATA = -32768
SCALE_FACTOR = 10_000
VALID_COUNT_BAND = "valid_count"

Sensor = Literal["Sentinel-2", "Landsat"]
Reducer = Literal["median", "mean", "min", "max"]
Band = Literal["blue", "green", "red", "nir", "swir1", "swir2"]
SpectralIndex = Literal["NDVI", "EVI", "SAVI", "MSAVI", "NDMI", "NDWI", "NBR", "NDBI", "NDSI", "BSI"]

STANDARD_BANDS: list[str] = ["blue", "green", "red", "nir", "swir1", "swir2"]

# Expressions are evaluated on surface reflectance (0-1), after compositing.
INDEX_EXPRESSIONS: dict[str, str] = {
    "NDVI": "(nir - red) / (nir + red)",
    "EVI": "2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
    "SAVI": "1.5 * (nir - red) / (nir + red + 0.5)",
    "MSAVI": "(2 * nir + 1 - sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2",
    "NDMI": "(nir - swir1) / (nir + swir1)",
    "NDWI": "(green - nir) / (green + nir)",
    "NBR": "(nir - swir2) / (nir + swir2)",
    "NDBI": "(swir1 - nir) / (swir1 + nir)",
    "NDSI": "(green - swir1) / (green + swir1)",
    "BSI": "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
}

_S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
_S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
# Scene classification classes kept as clear: vegetation, bare soil, water, unclassified, snow.
_S2_CLEAR_SCL = [4, 5, 6, 7, 11]

_LANDSAT_COLLECTIONS: list[tuple[str, list[str]]] = [
    ("LANDSAT/LT05/C02/T1_L2", ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]),
    ("LANDSAT/LE07/C02/T1_L2", ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]),
    ("LANDSAT/LC08/C02/T1_L2", ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]),
    ("LANDSAT/LC09/C02/T1_L2", ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]),
]
# QA_PIXEL bits 1-4: dilated cloud, cirrus, cloud, cloud shadow.
_LANDSAT_CLOUD_BITS = 0b11110

_DEFAULT_SCALE: dict[str, float] = {"Sentinel-2": 10.0, "Landsat": 30.0}

# computePixels rejects responses over ~48MB; stay well below it.
_MAX_TILE_BYTES = 32_000_000
_MAX_TILE_PX = 2048


def tile_size(n_bands: int, max_tile_px: int = _MAX_TILE_PX, max_tile_bytes: int = _MAX_TILE_BYTES) -> int:
    """Largest square tile edge (pixels) whose int16 payload fits under `max_tile_bytes`."""
    by_bytes = int(math.sqrt(max_tile_bytes / (n_bands * 2)))
    return max(1, min(max_tile_px, by_bytes))


def snapped_bounds(bounds: tuple[float, float, float, float], scale: float) -> tuple[float, float, float, float]:
    """Expand bounds outward so they fall on multiples of `scale`."""
    xmin, ymin, xmax, ymax = bounds
    return (
        math.floor(xmin / scale) * scale,
        math.floor(ymin / scale) * scale,
        math.ceil(xmax / scale) * scale,
        math.ceil(ymax / scale) * scale,
    )


def _prep_sentinel2(img):
    import ee

    clear = img.select("SCL").remap(_S2_CLEAR_SCL, [1] * len(_S2_CLEAR_SCL), 0)
    reflectance = img.select(_S2_BANDS, STANDARD_BANDS).divide(10_000).updateMask(clear)
    return ee.Image(reflectance.copyProperties(img, ["system:time_start"]))


def _landsat_prep(source_bands: list[str]):
    import ee

    def prep(img):
        clear = img.select("QA_PIXEL").bitwiseAnd(_LANDSAT_CLOUD_BITS).eq(0)
        unsaturated = img.select("QA_RADSAT").eq(0)
        reflectance = (
            img.select(source_bands, STANDARD_BANDS)
            .multiply(0.0000275)
            .add(-0.2)
            .updateMask(clear)
            .updateMask(unsaturated)
        )
        return ee.Image(reflectance.copyProperties(img, ["system:time_start"]))

    return prep


def clear_sky_collection(sensor: Sensor, geometry, start: str, end: str, cloud_threshold: float):
    """Scenes over `geometry` in [start, end) under the cloud threshold, cloud-masked, with standard band names."""
    import ee

    if sensor == "Sentinel-2":
        return (
            ee.ImageCollection(_S2_COLLECTION)
            .filterBounds(geometry)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
            .map(_prep_sentinel2)
        )

    merged = None
    for collection_id, source_bands in _LANDSAT_COLLECTIONS:
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(geometry)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUD_COVER", cloud_threshold))
            .map(_landsat_prep(source_bands))
        )
        merged = collection if merged is None else merged.merge(collection)
    return merged


def composite_image(collection, geometry, reducer: Reducer, bands: list[str], indices: list[str]):
    """Reduce `collection` to one int16 image: scaled bands, scaled indices, then `valid_count`."""
    import ee

    composite = getattr(collection, reducer)()
    band_images = {name: composite.select(name) for name in STANDARD_BANDS}

    layers = [composite.select(bands)] if bands else []
    layers += [composite.expression(INDEX_EXPRESSIONS[name], band_images).rename(name) for name in indices]
    scaled = ee.Image.cat(layers).multiply(SCALE_FACTOR).clamp(-32767, 32767).round().toInt16()

    valid_count = collection.select("red").count().unmask(0).toInt16().rename(VALID_COUNT_BAND)
    return ee.Image.cat([scaled, valid_count]).clip(geometry).unmask(NODATA, False)


def download_image(
    image,
    band_names: list[str],
    bounds: tuple[float, float, float, float],
    crs: str,
    scale: float,
    path: Path,
) -> None:
    """Fetch `image` tile by tile with computePixels and write a single int16 GeoTIFF."""
    import ee
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    xmin, ymin, xmax, ymax = bounds
    width = max(1, round((xmax - xmin) / scale))
    height = max(1, round((ymax - ymin) / scale))
    tile = tile_size(len(band_names))
    n_cols, n_rows = math.ceil(width / tile), math.ceil(height / tile)
    n_tiles = n_cols * n_rows
    print(
        f"Downloading {width} x {height} px ({len(band_names)} bands) as {n_tiles} tile(s) of up to {tile} px to {path}"
    )

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": len(band_names),
        "dtype": "int16",
        "crs": crs,
        "transform": from_origin(xmin, ymax, scale, scale),
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        done = 0
        for row_off in range(0, height, tile):
            for col_off in range(0, width, tile):
                window = Window(col_off, row_off, min(tile, width - col_off), min(tile, height - row_off))
                pixels = ee.data.computePixels(
                    {
                        "expression": image,
                        "fileFormat": "NUMPY_NDARRAY",
                        "grid": {
                            "dimensions": {"width": int(window.width), "height": int(window.height)},
                            "affineTransform": {
                                "scaleX": scale,
                                "shearX": 0,
                                "translateX": xmin + col_off * scale,
                                "shearY": 0,
                                "scaleY": -scale,
                                "translateY": ymax - row_off * scale,
                            },
                            "crsCode": crs,
                        },
                    }
                )
                dst.write(np.stack([pixels[name] for name in band_names]).astype("int16"), window=window)
                done += 1
                print(f"  tile {done}/{n_tiles} written")
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


@register()
def export_cloudfree_composites(
    client: EarthEngineClient,
    roi: AnyGeoDataFrame,
    time_range: Annotated[TimeRange, Field(description="Period to composite")],
    root_path: Annotated[str, Field(description="Directory the GeoTIFFs are written to")],
    sensor: Annotated[Sensor, Field(description="Satellite platform")] = "Sentinel-2",
    cloud_threshold: Annotated[
        float, Field(description="Maximum scene cloud cover (%) for a scene to be used", ge=0, le=100)
    ] = 60.0,
    reducer: Annotated[Reducer, Field(description="Statistic used to aggregate clear pixels over time")] = "median",
    bands: Annotated[list[Band], Field(description="Reflectance bands to include in the export")] = [
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
    ],
    indices: Annotated[list[SpectralIndex], Field(description="Spectral indices to add as bands")] = ["NDVI"],
    scale: Annotated[
        Optional[float],
        Field(description="Output pixel size in metres. Defaults to 10 (Sentinel-2) or 30 (Landsat)", gt=0),
    ] = None,
    crs: Annotated[
        Optional[str], Field(description="Output CRS, e.g. 'EPSG:32736'. Defaults to the ROI's UTM zone")
    ] = None,
) -> str:
    """Build one cloud-free composite over the ROI for the time range, write it as a GeoTIFF and return its path."""
    import ee

    if not bands and not indices:
        raise ValueError("Select at least one band or spectral index to export.")

    scale = scale or _DEFAULT_SCALE[sensor]
    crs = crs or roi.estimate_utm_crs().to_string()
    roi_shape = roi.to_crs(4326).union_all()
    geometry = ee.Geometry(roi_shape.__geo_interface__)
    bounds = snapped_bounds(tuple(roi.to_crs(crs).total_bounds), scale)
    band_names = [*bands, *indices, VALID_COUNT_BAND]
    roi_key = hashlib.md5(roi_shape.wkb).hexdigest()[:8]

    output_dir = Path(remove_file_scheme(str(root_path)))
    output_dir.mkdir(parents=True, exist_ok=True)

    label = f"{time_range.since:%Y-%m-%d}_{time_range.until:%Y-%m-%d}"
    print(f"Searching {sensor} scenes for {label} under {cloud_threshold}% cloud cover")
    collection = clear_sky_collection(
        sensor, geometry, time_range.since.isoformat(), time_range.until.isoformat(), cloud_threshold
    )
    image_count = collection.size().getInfo()
    if image_count == 0:
        raise ValueError(f"No {sensor} scenes under {cloud_threshold}% cloud cover for {label}.")
    print(f"Found {image_count} scene(s); building {reducer} composite with bands {band_names} at {scale} m in {crs}")

    image = composite_image(collection, geometry, reducer, list(bands), list(indices))
    path = output_dir / f"composite_{sensor.lower().replace('-', '')}_{roi_key}_{label}.tif"
    download_image(image, band_names, bounds, crs, scale, path)
    print(f"Composite written to {path}")
    return str(path)

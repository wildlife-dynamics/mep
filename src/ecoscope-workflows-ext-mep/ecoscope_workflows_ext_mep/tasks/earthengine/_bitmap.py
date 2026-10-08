"""Render a composite GeoTIFF as a map overlay image.

Polygonising a 10 m composite produces millions of features, so instead the raster
is reprojected to Web Mercator (what the map draws in), downsampled, coloured and
embedded as a PNG in a `BitmapLayerDefinition` for `merge_tile_layers` / the map task.
"""

import math
from typing import Annotated, Any, Literal

from ecoscope.platform.tasks.results._pydeck import BitmapLayerDefinition, LegendSegment, LegendValue
from pydantic import BaseModel, Discriminator, Field, Tag
from wt_registry import register

from ._composite import NODATA, SCALE_FACTOR, VALID_COUNT_BAND

# Band combinations rendered as RGB instead of through a palette.
COLOR_COMPOSITES: dict[str, list[str]] = {
    "true_color": ["red", "green", "blue"],
    "false_color": ["nir", "red", "green"],
}

# Continuous matplotlib colormaps that read well on rasters (no qualitative ones like tab10).
RasterColormap = Literal[
    # sequential, perceptually uniform
    "viridis",
    "cividis",
    "magma",
    "inferno",
    "plasma",
    # sequential, single hue / themed
    "Greens",
    "YlGn",
    "Blues",
    "YlGnBu",
    "Reds",
    "YlOrRd",
    "YlOrBr",
    "Greys",
    "gray",
    "bone",
    # diverging, for indices centred near zero
    "RdYlGn",
    "BrBG",
    "RdBu",
    "Spectral",
    "PiYG",
    "PuOr",
]


class ColormapPalette(BaseModel):
    type_: Literal["palette"] = "palette"
    name: Annotated[RasterColormap, Field(description="Matplotlib colormap name.")] = "viridis"
    reverse: Annotated[bool, Field(description="Flip the colormap so high values get the low-end colour.")] = False


class CustomPalette(BaseModel):
    type_: Literal["custom"] = "custom"
    colors: Annotated[list[str], Field(description="Hex colours from low to high values, e.g. ['#f7fcb9', '#31a354'].")]


def _palette_type(value: Any) -> str | None:
    """Pick the palette model, inferring it when `type_` is missing.

    `type_` has a default, so the form doesn't mark it required and params can arrive
    as just `{"name": "RdYlGn"}`. A plain `discriminator="type_"` rejects that, so fall
    back to the fields present: `colors` means a custom palette, anything else a colormap.
    """
    if isinstance(value, dict):
        return value.get("type_") or ("custom" if "colors" in value else "palette")
    return getattr(value, "type_", None)


ColorPalette = Annotated[
    Annotated[ColormapPalette, Tag("palette")] | Annotated[CustomPalette, Tag("custom")],
    Discriminator(_palette_type),
]

# Used when no palette is given: each band gets a colormap that suits what it measures.
DEFAULT_COLORMAPS: dict[str, ColormapPalette] = {
    "blue": ColormapPalette(name="Blues"),
    "green": ColormapPalette(name="Greens"),
    "red": ColormapPalette(name="Reds"),
    "nir": ColormapPalette(name="magma"),
    "swir1": ColormapPalette(name="inferno"),
    "swir2": ColormapPalette(name="cividis"),
    "NDVI": ColormapPalette(name="RdYlGn"),
    "EVI": ColormapPalette(name="RdYlGn"),
    "SAVI": ColormapPalette(name="YlGn"),
    "MSAVI": ColormapPalette(name="YlGn"),
    "NDMI": ColormapPalette(name="BrBG"),
    "NDWI": ColormapPalette(name="RdBu"),
    "NBR": ColormapPalette(name="RdYlGn"),
    "NDBI": ColormapPalette(name="YlOrRd"),
    "NDSI": ColormapPalette(name="bone"),
    "BSI": ColormapPalette(name="YlOrBr"),
    VALID_COUNT_BAND: ColormapPalette(name="viridis"),
}
FALLBACK_COLORMAP = ColormapPalette(name="viridis")
LEGEND_STEPS = 6


def _colormap(palette: ColormapPalette | CustomPalette, name: str):
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if isinstance(palette, CustomPalette):
        if not palette.colors:
            raise ValueError("A custom palette needs at least one hex colour.")
        return LinearSegmentedColormap.from_list(name, palette.colors)
    cmap = mpl.colormaps[palette.name]
    return cmap.reversed() if palette.reverse else cmap


def _read_web_mercator(src, band_names: list[str], max_size: int):
    """Reproject the named bands to EPSG:3857, at most `max_size` px on the long side.

    Returns unscaled float32 values with NaN for no data.
    """
    import numpy as np
    from affine import Affine
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    transform, width, height = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
    factor = max(width, height) / max_size
    if factor > 1:
        transform = transform * Affine.scale(factor)
        width, height = math.ceil(width / factor), math.ceil(height / factor)

    descriptions = list(src.descriptions)
    out = np.full((len(band_names), height, width), np.nan, dtype="float32")
    for i, name in enumerate(band_names):
        reproject(
            source=src.read(descriptions.index(name) + 1).astype("float32"),
            destination=out[i],
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=NODATA,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        if name != VALID_COUNT_BAND:
            out[i] /= SCALE_FACTOR
    return out, transform, width, height


def _stretch(values, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 2)) if vmin is None else vmin
    hi = float(np.percentile(finite, 98)) if vmax is None else vmax
    return (lo, hi) if hi > lo else (lo, lo + 1e-6)


def _lonlat_bounds(transform, width: int, height: int) -> list[float]:
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    west, south = to_lonlat.transform(left, bottom)
    east, north = to_lonlat.transform(right, top)
    return [west, south, east, north]


@register()
def composite_to_bitmap_layer(
    composite_path: Annotated[str, Field(description="GeoTIFF written by export_cloudfree_composites")],
    band: Annotated[
        str,
        Field(
            description="Band to show, e.g. 'NDVI', 'nir' or 'valid_count'; "
            "or 'true_color' (red/green/blue) / 'false_color' (nir/red/green)."
        ),
    ] = "NDVI",
    palette: Annotated[
        ColorPalette | None,
        Field(
            description="A named colormap or custom hex colours (single bands only). "
            "Defaults to a colormap suited to the band, e.g. RdYlGn for NDVI."
        ),
    ] = None,
    vmin: Annotated[
        float | None, Field(description="Value mapped to the first colour. Defaults to the 2nd percentile")
    ] = None,
    vmax: Annotated[
        float | None, Field(description="Value mapped to the last colour. Defaults to the 98th percentile")
    ] = None,
    opacity: Annotated[float, Field(description="Layer opacity", ge=0, le=1)] = 0.8,
    max_size: Annotated[int, Field(description="Longest side of the rendered image, in pixels", gt=0)] = 2048,
) -> BitmapLayerDefinition:
    """Colour one band (or an RGB band combination) of a composite and return it as a map overlay."""
    import base64
    import io

    import numpy as np
    import rasterio
    from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
    from matplotlib.colors import to_hex
    from PIL import Image

    path = remove_file_scheme(composite_path)
    band_names = COLOR_COMPOSITES.get(band, [band])
    with rasterio.open(path) as src:
        missing = [name for name in band_names if name not in src.descriptions]
        if missing:
            raise ValueError(f"Band(s) {missing} not in {path}. Available: {list(src.descriptions)}")
        print(f"Rendering {band} from {path} ({src.width} x {src.height} px)")
        data, transform, width, height = _read_web_mercator(src, band_names, max_size)

    valid = np.isfinite(data).all(axis=0)
    rgba = np.zeros((height, width, 4), dtype="uint8")
    legend = None

    if band in COLOR_COMPOSITES:
        for i, channel in enumerate(data):
            lo, hi = _stretch(channel, vmin, vmax)
            rgba[..., i] = (np.clip((np.nan_to_num(channel, nan=lo) - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
    else:
        cmap = _colormap(palette or DEFAULT_COLORMAPS.get(band, FALLBACK_COLORMAP), band)
        lo, hi = _stretch(data[0], vmin, vmax)
        normalised = np.clip((np.nan_to_num(data[0], nan=lo) - lo) / (hi - lo), 0, 1)
        rgba[..., :3] = (cmap(normalised)[..., :3] * 255).astype("uint8")
        fractions = [i / (LEGEND_STEPS - 1) for i in range(LEGEND_STEPS)]
        legend = LegendSegment(
            title=band,
            values=[LegendValue(label=f"{lo + (hi - lo) * f:.2f}", color=to_hex(cmap(f))) for f in fractions],
        )
    rgba[..., 3] = 255
    rgba[~valid] = 0

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    print(f"Rendered {width} x {height} px PNG ({buffer.tell() / 1e6:.1f} MB)")

    return BitmapLayerDefinition(
        image="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        bounds=_lonlat_bounds(transform, width, height),
        opacity=opacity,
        legend=legend,
    )

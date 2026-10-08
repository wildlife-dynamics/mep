"""Tests for ecoscope_workflows_ext_mep.tasks.earthengine._bitmap.

Renders a small synthetic UTM composite (written with rasterio into `tmp_path`)
and decodes the returned PNG to check size, transparency, colours and bounds.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import rasterio
from matplotlib import colormaps
from matplotlib.colors import to_hex
from PIL import Image
from pydantic import ValidationError
from rasterio.transform import from_origin

from ecoscope_workflows_ext_mep.tasks.earthengine._bitmap import (
    DEFAULT_COLORMAPS,
    ColormapPalette,
    CustomPalette,
    composite_to_bitmap_layer,
)
from ecoscope_workflows_ext_mep.tasks.earthengine._composite import NODATA, VALID_COUNT_BAND

BANDS = ["blue", "green", "red", "nir", "NDVI", VALID_COUNT_BAND]


@pytest.fixture
def composite_path(tmp_path) -> str:
    """40 x 30 px at 10 m in UTM 37S near Amboseli; NDVI ramps left to right, right-hand quarter is outside the ROI."""
    height, width = 30, 40
    ndvi = np.tile(np.linspace(1000, 8000, width, dtype="int16"), (height, 1))
    data = {
        "blue": np.full((height, width), 500, "int16"),
        "green": np.full((height, width), 800, "int16"),
        "red": np.full((height, width), 1000, "int16"),
        "nir": np.full((height, width), 3000, "int16"),
        "NDVI": ndvi,
        VALID_COUNT_BAND: np.full((height, width), 5, "int16"),
    }
    stack = np.stack([data[name] for name in BANDS])
    stack[:, :, 30:] = NODATA

    path = tmp_path / "composite.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=len(BANDS),
        dtype="int16",
        crs="EPSG:32737",
        transform=from_origin(300_000, 9_700_000, 10, 10),
        nodata=NODATA,
    ) as dst:
        dst.write(stack)
        for i, name in enumerate(BANDS, start=1):
            dst.set_band_description(i, name)
    return str(path)


def _decode(layer) -> np.ndarray:
    assert layer.image.startswith("data:image/png;base64,")
    png = base64.b64decode(layer.image.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


class TestCompositeToBitmapLayer:
    def test_single_band_defaults_to_the_bands_colormap(self, composite_path):
        layer = composite_to_bitmap_layer(composite_path, band="NDVI", vmin=0.0, vmax=1.0)

        rgba = _decode(layer)
        assert DEFAULT_COLORMAPS["NDVI"].name == "RdYlGn"
        assert layer.legend.title == "NDVI"
        assert [v.label for v in layer.legend.values] == ["0.00", "0.20", "0.40", "0.60", "0.80", "1.00"]
        assert layer.legend.values[0].color == to_hex(colormaps["RdYlGn"](0.0))
        assert layer.legend.values[-1].color == to_hex(colormaps["RdYlGn"](1.0))
        # NDVI increases left to right: red end on the left, green end on the right
        row = rgba[rgba.shape[0] // 2]
        opaque = row[row[:, 3] == 255]
        assert opaque[0, 0] > opaque[0, 1]
        assert opaque[-1, 1] > opaque[-1, 0]

    def test_named_colormap_can_be_reversed(self, composite_path):
        forward = composite_to_bitmap_layer(composite_path, palette=ColormapPalette(name="viridis"))
        reverse = composite_to_bitmap_layer(composite_path, palette=ColormapPalette(name="viridis", reverse=True))

        assert forward.legend.values[0].color == reverse.legend.values[-1].color
        assert forward.legend.values[-1].color == reverse.legend.values[0].color

    def test_custom_palette_spans_its_colours(self, composite_path):
        layer = composite_to_bitmap_layer(composite_path, palette=CustomPalette(colors=["#ffffff", "#000000"]))

        assert layer.legend.values[0].color == "#ffffff"
        assert layer.legend.values[-1].color == "#000000"

    def test_bands_without_a_default_fall_back_to_viridis(self, composite_path):
        layer = composite_to_bitmap_layer(composite_path, band="valid_count")

        assert layer.legend.values[-1].color == to_hex(colormaps["viridis"](1.0))

    def test_qualitative_colormaps_are_not_offered(self):
        with pytest.raises(ValidationError):
            ColormapPalette(name="tab10")

    def test_no_data_is_transparent(self, composite_path):
        rgba = _decode(composite_to_bitmap_layer(composite_path, band="NDVI"))

        mid = rgba.shape[0] // 2
        assert rgba[mid, 1, 3] == 255
        assert rgba[mid, -2, 3] == 0

    def test_bounds_are_lon_lat_around_the_raster(self, composite_path):
        west, south, east, north = composite_to_bitmap_layer(composite_path).bounds

        assert 37.1 < west < east < 37.3
        assert -2.8 < south < north < -2.6

    def test_image_is_capped_at_max_size(self, composite_path):
        rgba = _decode(composite_to_bitmap_layer(composite_path, max_size=10))

        assert max(rgba.shape[:2]) <= 10

    def test_true_color_has_no_legend(self, composite_path):
        layer = composite_to_bitmap_layer(composite_path, band="true_color", opacity=0.5)

        assert layer.legend is None
        assert layer.opacity == 0.5
        assert _decode(layer)[..., 3].max() == 255

    def test_unknown_band_lists_available_bands(self, composite_path):
        with pytest.raises(ValueError, match="Available"):
            composite_to_bitmap_layer(composite_path, band="EVI")

    def test_false_color_needs_its_bands(self, composite_path):
        layer = composite_to_bitmap_layer(composite_path, band="false_color")

        assert _decode(layer).shape[2] == 4

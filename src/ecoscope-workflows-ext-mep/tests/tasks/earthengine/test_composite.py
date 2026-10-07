"""Tests for ecoscope_workflows_ext_mep.tasks.earthengine.

Earth Engine needs credentials, so every server call is faked:
`ee.data.computePixels` is replaced with a stub that returns structured numpy
arrays shaped like the real NUMPY_NDARRAY response, and the collection /
composite builders are replaced in the export test. The GeoTIFF writing
runs for real against files in `tmp_path`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_mep.tasks.earthengine import _composite
from ecoscope_workflows_ext_mep.tasks.earthengine._composite import (
    NODATA,
    VALID_COUNT_BAND,
    download_image,
    export_cloudfree_composites,
    snapped_bounds,
    tile_size,
)

UTC = timezone.utc


def _dt(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _time_range(since: datetime, until: datetime) -> TimeRange:
    return TimeRange(
        since=since, until=until, timezone={"label": "UTC", "tzCode": "UTC", "name": "UTC", "utc": "+00:00"}
    )


class TestGridHelpers:
    def test_tile_size_respects_byte_budget(self):
        edge = tile_size(n_bands=8, max_tile_px=10_000, max_tile_bytes=1_000_000)

        assert edge * edge * 8 * 2 <= 1_000_000

    def test_tile_size_is_capped(self):
        assert tile_size(n_bands=1, max_tile_px=512) == 512

    def test_snapped_bounds_expand_to_the_grid(self):
        assert snapped_bounds((105.0, 201.0, 389.0, 410.0), 30.0) == (90.0, 180.0, 390.0, 420.0)


def _fake_compute_pixels(band_values: dict[str, int], calls: list[dict]):
    def compute_pixels(request):
        calls.append(request)
        dims = request["grid"]["dimensions"]
        out = np.zeros((dims["height"], dims["width"]), dtype=[(name, "<i2") for name in band_values])
        for name, value in band_values.items():
            out[name] = value
        return out

    return compute_pixels


class TestDownloadImage:
    def test_tiles_are_mosaicked_into_one_geotiff(self, tmp_path, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr("ee.data.computePixels", _fake_compute_pixels({"red": 1200, VALID_COUNT_BAND: 3}, calls))
        monkeypatch.setattr(_composite, "tile_size", lambda n_bands: 4)
        path = tmp_path / "out.tif"

        download_image("image", ["red", VALID_COUNT_BAND], (0.0, 0.0, 300.0, 180.0), "EPSG:32736", 30.0, path)

        with rasterio.open(path) as src:
            assert (src.width, src.height, src.count) == (10, 6, 2)
            assert src.descriptions == ("red", VALID_COUNT_BAND)
            assert src.nodata == NODATA
            assert (src.read(1) == 1200).all()
            assert (src.read(2) == 3).all()
        # 10x6 pixels in 4x4 tiles -> 3 columns x 2 rows
        assert len(calls) == 6
        last = calls[-1]["grid"]
        assert last["dimensions"] == {"width": 2, "height": 2}
        assert last["affineTransform"]["translateX"] == 240.0
        assert last["affineTransform"]["translateY"] == 60.0


class _FakeCollection:
    def __init__(self, n: int):
        self.n = n

    def size(self):
        return self

    def getInfo(self):
        return self.n


@pytest.fixture
def roi() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"name": ["a"]}, geometry=[box(36.0, -1.5, 36.01, -1.49)], crs=4326)


class TestExportCloudfreeComposites:
    @pytest.fixture
    def fake_ee(self, monkeypatch):
        state = {"image_count": 4, "downloads": []}

        monkeypatch.setattr("ee.Geometry", lambda geojson: geojson)
        monkeypatch.setattr(
            _composite,
            "clear_sky_collection",
            lambda sensor, geometry, start, end, threshold: _FakeCollection(state["image_count"]),
        )
        monkeypatch.setattr(_composite, "composite_image", lambda *args: "composite")
        monkeypatch.setattr(
            _composite,
            "download_image",
            lambda image, band_names, bounds, crs, scale, path: state["downloads"].append(
                (band_names, crs, scale, path)
            ),
        )
        return state

    def test_returns_the_written_geotiff_path(self, roi, tmp_path, fake_ee):
        result = export_cloudfree_composites(
            client=None,
            roi=roi,
            time_range=_time_range(_dt(2021, 1, 1), _dt(2022, 1, 1)),
            root_path=f"file://{tmp_path}",
            sensor="Landsat",
            bands=["red", "nir"],
            indices=["NDVI"],
        )

        [(band_names, crs, scale, path)] = fake_ee["downloads"]
        assert result == str(path)
        assert path.parent == tmp_path
        assert path.name.startswith("composite_landsat_") and path.name.endswith("_2021-01-01_2022-01-01.tif")
        assert band_names == ["red", "nir", "NDVI", VALID_COUNT_BAND]
        assert (crs, scale) == ("EPSG:32737", 30.0)

    def test_no_scenes_raises(self, roi, tmp_path, fake_ee):
        fake_ee["image_count"] = 0

        with pytest.raises(ValueError, match="No Sentinel-2 scenes"):
            export_cloudfree_composites(
                client=None, roi=roi, time_range=_time_range(_dt(2021, 1, 1), _dt(2022, 1, 1)), root_path=str(tmp_path)
            )
        assert fake_ee["downloads"] == []

    def test_requires_a_band_or_index(self, roi, tmp_path, fake_ee):
        with pytest.raises(ValueError, match="at least one band"):
            export_cloudfree_composites(
                client=None,
                roi=roi,
                time_range=_time_range(_dt(2021, 1, 1), _dt(2022, 1, 1)),
                root_path=str(tmp_path),
                bands=[],
                indices=[],
            )

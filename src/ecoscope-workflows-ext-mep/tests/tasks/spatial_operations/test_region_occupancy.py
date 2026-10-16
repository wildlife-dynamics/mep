"""Tests for ecoscope_workflows_ext_mep.tasks.spatial_operations._region_occupancy."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ecoscope_workflows_ext_mep.tasks.spatial_operations._region_occupancy import (
    OUTSIDE_REGIONS,
    compute_region_occupancy,
)


def _square(x0: float, y0: float, size: float = 1.0) -> Polygon:
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


class TestComputeRegionOccupancy:
    # Squares are laid out in meters around the central meridian of UTM zone 37S (near Nairobi),
    # the zone the task picks for them, so their areas come out exact.
    CRS = "EPSG:32737"

    def _sq(self, x, y, size=1000):
        return _square(500_000 + x, 9_850_000 + y, size)

    def _etd(self, *ranges, percentile=99.9, subjects=None):
        data = {"percentile": [percentile] * len(ranges)}
        if subjects is not None:
            data["subject_name"] = subjects
        return gpd.GeoDataFrame(data, geometry=list(ranges), crs=self.CRS)

    def _regions(self, names, geoms, column="name"):
        data = {} if column is None else {column: names}
        return gpd.GeoDataFrame(data, geometry=geoms, crs=self.CRS)

    def _run(self, etd, regions, **kwargs):
        return compute_region_occupancy(etd_gdf=etd, regions_gdf=regions, **kwargs).set_index("region")

    def test_percent_and_area_per_region_with_outside_row(self):
        regions = self._regions(["Park", "Ranch"], [self._sq(0, 0, 500), self._sq(5000, 5000)])

        result = self._run(self._etd(self._sq(0, 0)), regions)  # a 1 km² home range

        assert result.index.tolist() == ["Park", "Ranch", OUTSIDE_REGIONS]
        assert result.loc["Park", "occupancy_pct"] == pytest.approx(25.0)
        assert result.loc["Park", "area_km2"] == pytest.approx(0.25)
        assert result.loc["Park", "region_area_km2"] == pytest.approx(0.25)
        assert result.loc["Ranch", "region_area_km2"] == pytest.approx(1.0)
        assert pd.isna(result.loc[OUTSIDE_REGIONS, "region_area_km2"])
        assert result.loc["Ranch", "occupancy_pct"] == 0.0
        assert result.loc[OUTSIDE_REGIONS, "occupancy_pct"] == pytest.approx(75.0)
        assert result["home_range_km2"].tolist() == pytest.approx([1.0] * 3)

    def test_overlapping_regions_do_not_double_count_outside(self):
        regions = self._regions(["A", "B"], [self._sq(0, 0), self._sq(0, 0)])

        result = self._run(self._etd(self._sq(0, 0)), regions)

        assert result.loc["A", "occupancy_pct"] == result.loc["B", "occupancy_pct"] == 100.0
        assert result.loc[OUTSIDE_REGIONS, "occupancy_pct"] == 0.0

    def test_rows_with_the_same_region_name_are_merged(self):
        regions = self._regions(["Park", "Park"], [self._sq(0, 0, 500), self._sq(500, 500, 500)])

        result = self._run(self._etd(self._sq(0, 0)), regions)

        assert result.index.tolist() == ["Park", OUTSIDE_REGIONS]
        assert result.loc["Park", "occupancy_pct"] == pytest.approx(50.0)
        assert result.loc["Park", "region_area_km2"] == pytest.approx(0.5)

    def test_one_block_of_rows_per_subject(self):
        etd = self._etd(self._sq(0, 0), self._sq(2000, 0), subjects=["Luna", "Soutine"])

        result = self._run(etd, self._regions(["Park"], [self._sq(0, 0)])).loc["Park"]

        assert dict(zip(result["subject_name"], result["occupancy_pct"])) == {"Luna": 100.0, "Soutine": 0.0}

    def test_other_percentiles_are_ignored(self):
        etd = pd.concat([self._etd(self._sq(0, 0)), self._etd(self._sq(0, 0, 500), percentile=50.0)])

        result = self._run(etd, self._regions(["Park"], [self._sq(0, 0, 500)]), percentile=50.0)

        assert result.loc["Park", "occupancy_pct"] == pytest.approx(100.0)

    def test_areas_are_measured_in_the_local_utm_zone(self):
        # 0.1° squares in lon/lat: a true area of ~123.6 km² near the equator, and 25% overlap.
        etd = gpd.GeoDataFrame({"percentile": [99.9]}, geometry=[_square(36.0, -1.0, size=0.1)], crs="EPSG:4326")
        regions = gpd.GeoDataFrame({"name": ["Half"]}, geometry=[_square(36.0, -1.0, size=0.05)], crs="EPSG:4326")

        result = self._run(etd, regions)

        assert result.loc["Half", "occupancy_pct"] == pytest.approx(25.0, abs=0.2)
        assert result.loc["Half", "home_range_km2"] == pytest.approx(123.6, rel=0.005)

    def test_areas_stay_true_away_from_the_equator(self):
        # At 60°N a 0.1° square is half as wide (~61.9 km²); Web Mercator would report about 4x that.
        etd = gpd.GeoDataFrame({"percentile": [99.9]}, geometry=[_square(10.0, 60.0, size=0.1)], crs="EPSG:4326")
        regions = gpd.GeoDataFrame({"name": ["All"]}, geometry=[_square(10.0, 60.0, size=0.1)], crs="EPSG:4326")

        result = self._run(etd, regions)

        assert result.loc["All", "home_range_km2"] == pytest.approx(61.9, rel=0.01)

    def test_without_a_region_column_the_first_name_like_column_is_used(self):
        regions = self._regions(["Park"], [self._sq(0, 0, 500)], column="title")

        assert self._run(self._etd(self._sq(0, 0)), regions).index.tolist() == ["Park", OUTSIDE_REGIONS]

    def test_regions_without_any_name_column_are_numbered(self):
        regions = self._regions(None, [self._sq(0, 0, 500), self._sq(500, 0, 500)], column=None)

        result = self._run(self._etd(self._sq(0, 0)), regions)

        assert result.index.tolist() == ["Region 1", "Region 2", OUTSIDE_REGIONS]
        assert result["occupancy_pct"].tolist() == [25.0, 25.0, 50.0]

    def test_blank_names_are_numbered_not_merged(self):
        regions = self._regions(["Park", "", None], [self._sq(0, 0, 500), self._sq(500, 0, 500), self._sq(0, 500, 500)])

        result = self._run(self._etd(self._sq(0, 0)), regions)

        assert result.index.tolist() == ["Park", "Region 2", "Region 3", OUTSIDE_REGIONS]

    def test_missing_percentile_lists_the_available_ones(self):
        with pytest.raises(ValueError, match=r"no 99.9 percentile.*\[95.0\]"):
            self._run(self._etd(self._sq(0, 0), percentile=95.0), self._regions(["A"], [self._sq(0, 0)]))

    def test_missing_region_column_raises(self):
        with pytest.raises(ValueError, match="region column 'zone'"):
            self._run(self._etd(self._sq(0, 0)), self._regions(["A"], [self._sq(0, 0)]), region_column="zone")

    def test_empty_home_range_raises(self):
        with pytest.raises(ValueError, match="no area"):
            self._run(self._etd(Polygon()), self._regions(["A"], [self._sq(0, 0)]))

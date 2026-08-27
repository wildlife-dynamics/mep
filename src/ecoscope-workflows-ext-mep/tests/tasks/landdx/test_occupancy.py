"""Tests for ecoscope_workflows_ext_mep.tasks.landdx._occupancy.

All three functions (`build_ldx_template_region_lookup`,
`compute_template_regions`, `compute_subject_occupancy`) are registered via
`wt_registry.register()`, a no-op decorator at call time, so they are
exercised directly as plain Python against small, hand-built
GeoDataFrames -- no network or file I/O is involved in this module.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ecoscope_workflows_ext_mep.tasks.landdx._occupancy import (
    build_ldx_template_region_lookup,
    compute_subject_occupancy,
    compute_template_regions,
)


def _square(x0: float, y0: float, size: float = 1.0) -> Polygon:
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


# --------------------------------------------------------------------------- #
# build_ldx_template_region_lookup                                            #
# --------------------------------------------------------------------------- #


class TestBuildLdxTemplateRegionLookup:
    def test_groups_globalids_by_category(self):
        gdf = gpd.GeoDataFrame(
            {
                "type": ["National Park", "Community Conservancy", "Group Ranch", "Unmapped Type"],
                "globalid": ["np-1", "cc-1", "gr-1", "other-1"],
                "geometry": [_square(0, 0), _square(1, 1), _square(2, 2), _square(3, 3)],
            },
            crs="EPSG:4326",
        )

        result = build_ldx_template_region_lookup(gdf)

        assert result["national_pa_use"] == ["np-1"]
        assert sorted(result["community_pa_use"]) == ["cc-1", "gr-1"]

    def test_includes_hardcoded_static_ids(self):
        gdf = gpd.GeoDataFrame({"type": [], "globalid": [], "geometry": []}, crs="EPSG:4326")

        result = build_ldx_template_region_lookup(gdf)

        assert result["crop_raid_percent"] == ["2d3f6392-700c-495f-8bc5-087538f6f125"]
        assert result["kenya_use"] == ["7895ded1-df29-4ca1-8e34-ebc8e3cbb24e"]

    def test_rows_with_empty_geometry_are_excluded(self):
        gdf = gpd.GeoDataFrame(
            {
                "type": ["National Park", "National Park"],
                "globalid": ["np-1", "np-empty"],
                "geometry": [_square(0, 0), Polygon()],
            },
            crs="EPSG:4326",
        )

        result = build_ldx_template_region_lookup(gdf)

        assert result["national_pa_use"] == ["np-1"]

    def test_rows_with_null_type_or_globalid_are_excluded(self):
        gdf = gpd.GeoDataFrame(
            {
                "type": ["National Park", None],
                "globalid": [None, "np-2"],
                "geometry": [_square(0, 0), _square(1, 1)],
            },
            crs="EPSG:4326",
        )
        # Give the first row (null globalid) a valid type but no id, and the
        # second row a null type but a valid id -- neither should surface.
        result = build_ldx_template_region_lookup(gdf)

        assert result["national_pa_use"] == []


# --------------------------------------------------------------------------- #
# compute_template_regions                                                    #
# --------------------------------------------------------------------------- #


class TestComputeTemplateRegions:
    def test_unions_matching_geometries_per_template(self):
        gdf = gpd.GeoDataFrame(
            {
                "globalid": ["a", "b", "c"],
                "geometry": [_square(0, 0), _square(0.5, 0.5), _square(10, 10)],
            },
            crs="EPSG:4326",
        )
        template_lookup = {"national_pa_use": ["a", "b"], "community_pa_use": ["c"]}

        result = compute_template_regions(gdf, template_lookup, crs="EPSG:4326")

        assert set(result.keys()) == {"national_pa_use", "community_pa_use"}
        # "a" and "b" overlap, so their union area is less than the sum of
        # the two individual 1x1 squares (2.0) but at least as large as one.
        assert 1.0 <= result["national_pa_use"].area < 2.0
        assert result["community_pa_use"].area == pytest.approx(1.0)

    def test_no_matching_globalids_returns_empty_polygon(self):
        gdf = gpd.GeoDataFrame(
            {"globalid": ["a"], "geometry": [_square(0, 0)]},
            crs="EPSG:4326",
        )
        template_lookup = {"national_pa_use": ["does-not-exist"]}

        result = compute_template_regions(gdf, template_lookup, crs="EPSG:4326")

        assert result["national_pa_use"].is_empty

    def test_reprojects_to_requested_crs(self):
        gdf = gpd.GeoDataFrame(
            {"globalid": ["a"], "geometry": [_square(36.0, -1.0)]},
            crs="EPSG:4326",
        )
        template_lookup = {"national_pa_use": ["a"]}

        result = compute_template_regions(gdf, template_lookup, crs="EPSG:32737")

        # A ~1-degree-square polygon reprojected to a metric CRS has an area
        # on the order of 1e10 m^2, nowhere near the 1.0 deg^2 original.
        assert result["national_pa_use"].area > 1000.0


# --------------------------------------------------------------------------- #
# compute_subject_occupancy                                                   #
# --------------------------------------------------------------------------- #


class TestComputeSubjectOccupancy:
    def _etd_gdf(self, geom, percentile=99.9, crs="EPSG:32737"):
        return gpd.GeoDataFrame({"percentile": [percentile]}, geometry=[geom], crs=crs)

    def test_computes_percent_occupancy_per_region(self):
        subjects_df = pd.DataFrame({"subject_name": ["Cherop"]})
        home_range = _square(0, 0, size=10)  # area 100
        etd_gdf = self._etd_gdf(home_range)
        regions_gdf = {
            "national_pa_use": _square(0, 0, size=5),  # 25 / 100 = 25%
            "community_pa_use": Polygon(),  # no overlap
        }

        result = compute_subject_occupancy(
            subjects_df=subjects_df, crs="EPSG:32737", etd_gdf=etd_gdf, regions_gdf=regions_gdf
        )

        row = result.iloc[0]
        assert row["national_pa_use"] == pytest.approx(25.0)
        assert row["community_pa_use"] == pytest.approx(0.0)
        assert row["unprotected"] == pytest.approx(75.0)

    def test_unprotected_never_goes_below_zero(self):
        subjects_df = pd.DataFrame({"subject_name": ["Cherop"]})
        home_range = _square(0, 0, size=10)  # area 100
        etd_gdf = self._etd_gdf(home_range)
        # Two overlapping regions that together exceed 100% coverage.
        regions_gdf = {
            "national_pa_use": _square(0, 0, size=10),
            "community_pa_use": _square(0, 0, size=10),
        }

        result = compute_subject_occupancy(
            subjects_df=subjects_df, crs="EPSG:32737", etd_gdf=etd_gdf, regions_gdf=regions_gdf
        )

        assert result.iloc[0]["unprotected"] == pytest.approx(0.0)

    def test_missing_99_9_percentile_raises(self):
        subjects_df = pd.DataFrame({"subject_name": ["Cherop"]})
        etd_gdf = self._etd_gdf(_square(0, 0, size=10), percentile=95.0)

        with pytest.raises(ValueError, match="No 99.9th percentile"):
            compute_subject_occupancy(subjects_df=subjects_df, crs="EPSG:32737", etd_gdf=etd_gdf, regions_gdf={})

    def test_empty_home_range_geometry_raises(self):
        subjects_df = pd.DataFrame({"subject_name": ["Cherop"]})
        etd_gdf = self._etd_gdf(Polygon())

        with pytest.raises(ValueError, match="Home range geometry is empty"):
            compute_subject_occupancy(subjects_df=subjects_df, crs="EPSG:32737", etd_gdf=etd_gdf, regions_gdf={})

    def test_zero_area_home_range_raises(self):
        subjects_df = pd.DataFrame({"subject_name": ["Cherop"]})
        # A degenerate (collinear-vertex) polygon: non-empty, but zero area.
        degenerate = Polygon([(0, 0), (1, 0), (2, 0), (0, 0)])
        etd_gdf = self._etd_gdf(degenerate)

        with pytest.raises(ValueError, match="zero area"):
            compute_subject_occupancy(subjects_df=subjects_df, crs="EPSG:32737", etd_gdf=etd_gdf, regions_gdf={})

    def test_region_intersection_error_falls_back_to_zero(self):
        subjects_df = pd.DataFrame({"subject_name": ["Cherop"]})
        etd_gdf = self._etd_gdf(_square(0, 0, size=10))
        regions_gdf = {"national_pa_use": "not-a-geometry"}

        result = compute_subject_occupancy(
            subjects_df=subjects_df, crs="EPSG:32737", etd_gdf=etd_gdf, regions_gdf=regions_gdf
        )

        assert result.iloc[0]["national_pa_use"] == 0.0

    def test_values_are_rounded_to_one_decimal(self):
        subjects_df = pd.DataFrame({"subject_name": ["Cherop"]})
        home_range = _square(0, 0, size=3)  # area 9
        etd_gdf = self._etd_gdf(home_range)
        regions_gdf = {"national_pa_use": _square(0, 0, size=1)}  # 1/9 = 11.111...%

        result = compute_subject_occupancy(
            subjects_df=subjects_df, crs="EPSG:32737", etd_gdf=etd_gdf, regions_gdf=regions_gdf
        )

        value = result.iloc[0]["national_pa_use"]
        assert value == round(value, 1)

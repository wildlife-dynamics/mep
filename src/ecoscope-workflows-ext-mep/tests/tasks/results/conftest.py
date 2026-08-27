"""Shared fixtures for ecoscope_workflows_ext_mep.tasks.results tests.

Loads the real `relocs.parquet` fixture (a year of hourly real EarthRanger
relocations for one subject, "Esposito") once per session, and derives
`Relocations`/`Trajectory` objects from it via the real `ecoscope` package
rather than mocking trajectory computation, since it is installed and
cheap to run for real.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from ecoscope.relocations import Relocations
from ecoscope.trajectory import Trajectory

DATA_DIR = Path(__file__).parents[3] / "tests" / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def _relocs_raw() -> gpd.GeoDataFrame:
    """A year of real, hourly relocations for one subject ("Esposito"),
    reshaped into RelocationsGDFSchema-like columns.
    """
    gdf = gpd.read_parquet(DATA_DIR / "relocs.parquet")
    gdf = gdf.rename(columns={"extra__subject__name": "subject_name"})
    return gdf[["groupby_col", "fixtime", "geometry", "subject_name", "junk_status"]].reset_index(drop=True)


@pytest.fixture
def relocs_gdf(_relocs_raw) -> gpd.GeoDataFrame:
    """8758 hourly relocation points for subject 'Esposito' over 2024, EPSG:4326.

    Columns: groupby_col, fixtime, geometry, subject_name, junk_status.
    """
    return _relocs_raw.copy()


@pytest.fixture(scope="session")
def _traj_raw(_relocs_raw) -> gpd.GeoDataFrame:
    relocations = Relocations.from_gdf(_relocs_raw.copy())
    trajectory = Trajectory.from_relocations(relocations)
    return trajectory.gdf


@pytest.fixture
def traj_gdf(_traj_raw) -> gpd.GeoDataFrame:
    """Trajectory segments derived from `relocs_gdf` via `Trajectory.from_relocations`.

    Columns include: groupby_col, segment_start, segment_end, dist_meters,
    speed_kmhr, heading, timespan_seconds, nsd, extra__subject_name, geometry.
    """
    return _traj_raw.copy()


@pytest.fixture
def seasons_df() -> pd.DataFrame:
    """Two non-overlapping seasonal windows spanning all of 2024."""
    return pd.DataFrame(
        {
            "start": ["2024-01-01", "2024-07-01"],
            "end": ["2024-06-30", "2024-12-31"],
            "season": ["dry", "wet"],
        }
    )

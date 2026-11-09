"""Shared fixtures for the animate tests."""

from __future__ import annotations

import pytest

from ecoscope_workflows_ext_mep.tasks.animate import _draw_map as dm

_REAL_TERRAIN_GRID = dm._terrain_grid  # captured before any test patches it


@pytest.fixture(autouse=True)
def no_terrain_grid_downloads(monkeypatch):
    """draw_animated_map samples elevation tiles for the camera's line-of-sight grid; tests don't go
    online for it. Tests of the grid itself take the `real_terrain_grid` fixture."""
    monkeypatch.setattr(dm, "_terrain_grid", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def results_folder(monkeypatch, tmp_path):
    """Camera settings and waypoint files are only read from inside the results folder; tests write
    theirs to tmp_path, so that is the results folder."""
    monkeypatch.setenv("ECOSCOPE_WORKFLOWS_RESULTS", str(tmp_path))
    return tmp_path


@pytest.fixture
def real_terrain_grid():
    return _REAL_TERRAIN_GRID

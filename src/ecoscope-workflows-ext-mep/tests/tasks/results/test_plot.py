"""Tests for ecoscope_workflows_ext_mep.tasks.results._plot.

`draw_season_nsd_plot`, `draw_season_speed_plot`, `draw_season_mcp_plot`,
and `draw_season_collared_plot` are registered via `wt_registry.register()`,
a no-op decorator at call time, so they are called directly as plain
Python. They are exercised here against the real, year-long single-subject
relocations fixture (see `conftest.py`) run through the real `ecoscope`
`Relocations`/`Trajectory` machinery -- nothing here is mocked, since it's
all installed and cheap to run for real.

The module-level `add_seasons_square` and `nsd` helpers (not registered)
are also covered directly, since most of the interesting edge-case
behavior (skipping unparsable/invalid season rows, the "no data" branch)
lives there.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go
import pytest

from ecoscope_workflows_ext_mep.tasks.results._plot import (
    add_seasons_square,
    draw_season_collared_plot,
    draw_season_mcp_plot,
    draw_season_nsd_plot,
    draw_season_speed_plot,
    nsd,
)


# --------------------------------------------------------------------------- #
# add_seasons_square                                                          #
# --------------------------------------------------------------------------- #


class TestAddSeasonsSquare:
    def test_draws_one_shape_per_valid_row(self):
        season_df = pd.DataFrame(
            {"start": ["2024-01-01", "2024-07-01"], "end": ["2024-06-30", "2024-12-31"], "season": ["dry", "wet"]}
        )

        fig = add_seasons_square(go.Figure(), season_df)

        assert len(fig.layout.shapes) == 2

    def test_skips_rows_with_unparsable_dates(self):
        season_df = pd.DataFrame({"start": ["not-a-date"], "end": ["2024-06-30"], "season": ["dry"]})

        fig = add_seasons_square(go.Figure(), season_df)

        assert len(fig.layout.shapes) == 0

    def test_skips_rows_where_end_is_not_after_start(self):
        season_df = pd.DataFrame({"start": ["2024-08-01"], "end": ["2024-07-01"], "season": ["dry"]})

        fig = add_seasons_square(go.Figure(), season_df)

        assert len(fig.layout.shapes) == 0

    def test_one_legend_entry_per_unique_season_type(self):
        season_df = pd.DataFrame(
            {
                "start": ["2024-01-01", "2024-03-01", "2024-07-01"],
                "end": ["2024-02-01", "2024-04-01", "2024-08-01"],
                "season": ["dry", "dry", "wet"],
            }
        )

        fig = add_seasons_square(go.Figure(), season_df, show_legend=True)

        assert len(fig.data) == 2  # one dummy trace per unique season, not per row

    def test_show_legend_false_adds_no_legend_traces(self):
        season_df = pd.DataFrame({"start": ["2024-01-01"], "end": ["2024-02-01"], "season": ["dry"]})

        fig = add_seasons_square(go.Figure(), season_df, show_legend=False)

        assert len(fig.data) == 0

    def test_unmapped_season_uses_fallback_color(self):
        season_df = pd.DataFrame({"start": ["2024-01-01"], "end": ["2024-02-01"], "season": ["unmapped"]})

        fig = add_seasons_square(go.Figure(), season_df)

        assert fig.layout.shapes[0].fillcolor == "rgba(120, 120, 120, 0.10)"

    def test_colors_override_replaces_default_for_matching_key(self):
        season_df = pd.DataFrame({"start": ["2024-01-01"], "end": ["2024-02-01"], "season": ["dry"]})

        fig = add_seasons_square(go.Figure(), season_df, colors={"dry": "rgba(1,2,3,0.5)"})

        assert fig.layout.shapes[0].fillcolor == "rgba(1,2,3,0.5)"


# --------------------------------------------------------------------------- #
# nsd (helper)                                                                #
# --------------------------------------------------------------------------- #


class TestNsd:
    def test_empty_relocations_returns_annotated_empty_figure(self, relocs_gdf):
        empty = relocs_gdf.iloc[0:0]

        fig = nsd(empty)

        assert fig.layout.annotations[0].text == "No relocation data"
        assert len(fig.data) == 0

    def test_one_trace_per_subject(self, relocs_gdf):
        fig = nsd(relocs_gdf)

        assert len(fig.data) == relocs_gdf["groupby_col"].nunique()

    def test_unknown_legend_column_falls_back_to_groupby_col(self, relocs_gdf):
        fig = nsd(relocs_gdf, legend_column="does_not_exist")

        assert len(fig.data) == 1
        assert fig.data[0].name == relocs_gdf["groupby_col"].iloc[0]

    def test_legend_column_used_when_present(self, relocs_gdf):
        fig = nsd(relocs_gdf, legend_column="subject_name")

        assert fig.data[0].name == "Esposito"


# --------------------------------------------------------------------------- #
# draw_season_nsd_plot                                                        #
# --------------------------------------------------------------------------- #


class TestDrawSeasonNsdPlot:
    def test_returns_non_empty_html(self, relocs_gdf, seasons_df):
        html = draw_season_nsd_plot(relocations_gdf=relocs_gdf, seasons_df=seasons_df)

        assert isinstance(html, str)
        assert "<div" in html

    def test_widget_id_is_embedded_in_html(self, relocs_gdf, seasons_df):
        html = draw_season_nsd_plot(relocations_gdf=relocs_gdf, seasons_df=seasons_df, widget_id="my-widget")

        assert "my-widget" in html


# --------------------------------------------------------------------------- #
# draw_season_speed_plot                                                      #
# --------------------------------------------------------------------------- #


class TestDrawSeasonSpeedPlot:
    def test_returns_non_empty_html(self, relocs_gdf, seasons_df):
        html = draw_season_speed_plot(relocations_gdf=relocs_gdf, seasons_df=seasons_df)

        assert isinstance(html, str)
        assert "<div" in html


# --------------------------------------------------------------------------- #
# draw_season_mcp_plot                                                        #
# --------------------------------------------------------------------------- #


class TestDrawSeasonMcpPlot:
    def test_returns_non_empty_html(self, relocs_gdf, seasons_df):
        html = draw_season_mcp_plot(relocations_gdf=relocs_gdf, seasons_df=seasons_df)

        assert isinstance(html, str)
        assert "<div" in html

    def test_empty_relocations_raises(self, relocs_gdf, seasons_df):
        with pytest.raises(ValueError, match="Relocations gdf is empty"):
            draw_season_mcp_plot(relocations_gdf=relocs_gdf.iloc[0:0], seasons_df=seasons_df)

    def test_none_relocations_raises(self, seasons_df):
        with pytest.raises(ValueError, match="Relocations gdf is empty"):
            draw_season_mcp_plot(relocations_gdf=None, seasons_df=seasons_df)


# --------------------------------------------------------------------------- #
# draw_season_collared_plot                                                   #
# --------------------------------------------------------------------------- #


class TestDrawSeasonCollaredPlot:
    def test_returns_non_empty_html_with_matching_events(self, relocs_gdf, seasons_df):
        events_gdf = pd.DataFrame(
            {
                "subject_name": ["Esposito"],
                "time": pd.to_datetime(["2024-03-01T00:00:00Z"]),
                "priority_label": ["red"],
                "event_type": ["mep_collaring"],
            }
        )

        html = draw_season_collared_plot(events_gdf=events_gdf, relocations_gdf=relocs_gdf, seasons_df=seasons_df)

        assert isinstance(html, str)
        assert "<div" in html

    def test_none_events_still_returns_html(self, relocs_gdf, seasons_df):
        html = draw_season_collared_plot(events_gdf=None, relocations_gdf=relocs_gdf, seasons_df=seasons_df)

        assert isinstance(html, str)
        assert "<div" in html

    def test_events_for_other_subjects_are_filtered_out(self, relocs_gdf, seasons_df):
        events_gdf = pd.DataFrame(
            {
                "subject_name": ["SomeoneElse"],
                "time": pd.to_datetime(["2024-03-01T00:00:00Z"]),
                "priority_label": ["red"],
                "event_type": ["mep_collaring"],
            }
        )

        # Should not raise even though no events match this subject.
        html = draw_season_collared_plot(events_gdf=events_gdf, relocations_gdf=relocs_gdf, seasons_df=seasons_df)

        assert isinstance(html, str)

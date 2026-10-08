"""Tests for ecoscope_workflows_ext_mep.tasks.results._subject_source.

`draw_subject_source_diagram` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python
against small hand-built subjects/sources/subjectsources frames shaped like
EarthRanger's responses.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
from pydantic import ValidationError

from ecoscope_workflows_ext_mep.tasks.results._subject_source import (
    DiagramLayoutStyle,
    LinkStyle,
    NodeStyle,
    build_subject_source_graph,
    draw_subject_source_diagram,
    layout_subject_source_diagram,
    link_colors,
    node_colors,
)


@pytest.fixture
def subjects():
    return pd.DataFrame(
        [
            {
                "id": "amani",
                "name": "Amani",
                "subject_type": "wildlife",
                "is_active": True,
            },
            {
                "id": "baraka",
                "name": "Baraka",
                "subject_type": "wildlife",
                "is_active": True,
            },
            {
                "id": "zawadi",
                "name": "Zawadi",
                "subject_type": "wildlife",
                "is_active": False,
            },
            {
                "id": "lonely",
                "name": "Lonely",
                "subject_type": "wildlife",
                "is_active": True,
            },
        ]
    )


@pytest.fixture
def sources():
    return pd.DataFrame(
        [
            {"id": "s1", "manufacturer_id": "ST-1001", "provider": "savannah"},
            {"id": "s0", "manufacturer_id": "ST-0950", "provider": "savannah"},
            {"id": "s2", "manufacturer_id": "ST-1002", "provider": "savannah"},
            {"id": "a1", "manufacturer_id": "AWT-2001", "provider": "awt"},
            {"id": "spare", "manufacturer_id": "ST-9999", "provider": "savannah"},
        ]
    )


@pytest.fixture
def subjectsources():
    # Raw ER shape: assigned_range is a {lower, upper} dict, 9999 means open-ended.
    return pd.DataFrame(
        [
            {
                "subject": "amani",
                "source": "s1",
                "assigned_range": {
                    "lower": "2024-03-01T00:00:00+00:00",
                    "upper": "9999-12-31T23:59:59+00:00",
                },
            },
            {
                "subject": "amani",
                "source": "s0",
                "assigned_range": {
                    "lower": "2022-01-15T00:00:00+00:00",
                    "upper": "2024-02-28T00:00:00+00:00",
                },
            },
            {
                "subject": "baraka",
                "source": "s2",
                "assigned_range": {"lower": "0023-06-10T00:00:00+00:00", "upper": None},
            },
            {
                "subject": "zawadi",
                "source": "a1",
                "assigned_range": {
                    "lower": "2021-05-01T00:00:00+00:00",
                    "upper": "2022-11-30T00:00:00+00:00",
                },
            },
        ]
    )


def _edge(edges, subject, source):
    return next(
        e
        for e in edges
        if e["from"] == f"subject:{subject}" and e["to"] == f"source:{source}"
    )


def _node(nodes, node_id):
    return next(n for n in nodes if n["id"] == node_id)


class TestBuildSubjectSourceGraph:
    def test_one_node_per_subject_and_source(self, subjects, subjectsources, sources):
        nodes, edges = build_subject_source_graph(subjects, subjectsources, sources)
        assert len([n for n in nodes if n["group"] == "subject"]) == 4
        assert len([n for n in nodes if n["group"] == "source"]) == 5
        assert len(edges) == 4

    def test_current_assignment_is_solid_and_open_ended(
        self, subjects, subjectsources, sources
    ):
        _, edges = build_subject_source_graph(subjects, subjectsources, sources)
        edge = _edge(edges, "amani", "s1")
        assert edge["label"] == "2024-03-01 → present"
        assert edge["current"] is True

    def test_ended_assignment_is_dashed(self, subjects, subjectsources, sources):
        _, edges = build_subject_source_graph(subjects, subjectsources, sources)
        edge = _edge(edges, "amani", "s0")
        assert edge["label"] == "2022-01-15 → 2024-02-28"
        assert edge["current"] is False

    def test_out_of_range_date_is_flagged(self, subjects, subjectsources, sources):
        _, edges = build_subject_source_graph(subjects, subjectsources, sources)
        edge = _edge(edges, "baraka", "s2")
        assert edge["label"].startswith("⚠ 0023-06-10")
        assert edge["bad_date"] is True

    def test_flattened_range_columns(self, subjects, sources):
        flat = pd.DataFrame(
            [
                {
                    "subject": "amani",
                    "source": "s1",
                    "assigned_range_lower": pd.Timestamp("2024-03-01", tz="UTC"),
                    "assigned_range_upper": pd.NaT,
                    "invalid_assigned_range_lower": None,
                    "invalid_assigned_range_upper": None,
                },
                {
                    "subject": "baraka",
                    "source": "s2",
                    "assigned_range_lower": pd.NaT,
                    "assigned_range_upper": pd.NaT,
                    "invalid_assigned_range_lower": "0023-06-10T00:00:00+00:00",
                    "invalid_assigned_range_upper": None,
                },
            ]
        )
        _, edges = build_subject_source_graph(subjects, flat, sources)
        assert _edge(edges, "amani", "s1")["label"] == "2024-03-01 → present"
        assert _edge(edges, "baraka", "s2")["bad_date"] is True

    def test_unlinked_subject_and_source_are_flagged(
        self, subjects, subjectsources, sources
    ):
        nodes, _ = build_subject_source_graph(subjects, subjectsources, sources)
        assert _node(nodes, "subject:lonely")["linked"] is False
        assert _node(nodes, "source:spare")["linked"] is False
        assert _node(nodes, "source:s1")["linked"] is True

    def test_can_hide_unlinked_sources(self, subjects, subjectsources, sources):
        nodes, _ = build_subject_source_graph(
            subjects, subjectsources, sources, include_unlinked_sources=False
        )
        assert "source:spare" not in {n["id"] for n in nodes}

    def test_numpy_bool_inactive_subject_is_flagged(
        self, subjects, subjectsources, sources
    ):
        nodes, _ = build_subject_source_graph(subjects, subjectsources, sources)
        assert _node(nodes, "subject:zawadi")["inactive"] is True
        assert _node(nodes, "subject:amani")["inactive"] is False
        assert "inactive" in _node(nodes, "subject:zawadi")["title"]

    def test_assignments_for_other_groups_are_dropped(
        self, subjects, subjectsources, sources
    ):
        _, edges = build_subject_source_graph(
            subjects[subjects["id"] == "amani"], subjectsources, sources
        )
        assert {e["from"] for e in edges} == {"subject:amani"}

    def test_assignment_to_unknown_source_still_drawn(self, subjects, sources):
        links = pd.DataFrame(
            [{"subject": "amani", "source": "ghost-source-id", "assigned_range": None}]
        )
        nodes, edges = build_subject_source_graph(subjects, links, sources)
        assert "not found in sources" in _node(nodes, "source:ghost-source-id")["title"]
        assert _node(nodes, "source:ghost-source-id")["missing"] is True
        assert len(edges) == 1


class TestLayout:
    def _layout(self, subjects, subjectsources, sources):
        return layout_subject_source_diagram(
            *build_subject_source_graph(subjects, subjectsources, sources)
        )

    def test_subjects_alphabetical_with_one_row_per_assignment(
        self, subjects, subjectsources, sources
    ):
        layout = self._layout(subjects, subjectsources, sources)
        linked = [n for n in layout["subjects"] if n["row"] < layout["unlinked_row"]]
        assert [n["label"] for n in linked] == ["Amani", "Baraka", "Zawadi"]
        # Amani has two assignments, so it spans rows 0-1 and sits between them.
        assert linked[0]["row"] == 0.5
        assert sorted(link["row"] for link in layout["links"]) == [0, 1, 2, 3]

    def test_sources_sit_beside_their_assignments(
        self, subjects, subjectsources, sources
    ):
        layout = self._layout(subjects, subjectsources, sources)
        link_rows = {link["to"]: link["row"] for link in layout["links"]}
        for node in layout["sources"]:
            if node["id"] in link_rows:
                assert node["row"] == link_rows[node["id"]]

    def test_shared_source_does_not_overlap(self, subjects, sources):
        links = pd.DataFrame(
            [
                {"subject": "amani", "source": "s1", "assigned_range": None},
                {"subject": "baraka", "source": "s1", "assigned_range": None},
                {"subject": "baraka", "source": "s2", "assigned_range": None},
                {"subject": "zawadi", "source": "s0", "assigned_range": None},
            ]
        )
        layout = self._layout(subjects, links, sources)
        rows = [n["row"] for n in layout["sources"]]
        assert len(rows) == len(set(rows))
        assert all(b - a >= 1 for a, b in zip(sorted(rows), sorted(rows)[1:]))

    def test_unlinked_listed_at_bottom(self, subjects, subjectsources, sources):
        layout = self._layout(subjects, subjectsources, sources)
        below = layout["unlinked_row"]
        assert [n["label"] for n in layout["subjects"] if n["row"] > below] == [
            "Lonely"
        ]
        assert [n["label"] for n in layout["sources"] if n["row"] > below] == [
            "ST-9999"
        ]
        assert all(link["row"] < below for link in layout["links"])

    def test_no_unlinked_section_when_everything_is_linked(
        self, subjects, subjectsources, sources
    ):
        layout = self._layout(
            subjects[subjects["id"] != "lonely"],
            subjectsources,
            sources[sources["id"] != "spare"],
        )
        assert layout["unlinked_row"] is None


class TestDrawSubjectSourceDiagram:
    def test_returns_static_svg_html(self, subjects, subjectsources, sources):
        html = draw_subject_source_diagram(
            subjects=subjects,
            subjectsources=subjectsources,
            sources=sources,
            widget_id="Subject-source links",
        )
        assert '<svg id="Subject-source-links"' in html
        assert "<script src=" not in html  # no CDN dependency
        assert html.count('class="el node') == 9
        assert "2024-03-01 → present" in html
        assert "Subjects with no source" in html

    def test_height_grows_with_rows(self, subjects, subjectsources):
        many_sources = pd.DataFrame(
            [{"id": f"x{i}", "manufacturer_id": f"X-{i}"} for i in range(300)]
        )
        html = draw_subject_source_diagram(
            subjects=subjects,
            subjectsources=subjectsources,
            sources=many_sources,
            row_height=20,
        )
        height = int(re.search(r'<svg id=[^>]*height="(\d+)"', html).group(1))
        assert height > 300 * 20

    def test_empty_inputs_render_message(self):
        empty = pd.DataFrame(columns=["id"])
        html = draw_subject_source_diagram(
            subjects=empty,
            subjectsources=pd.DataFrame(columns=["subject", "source"]),
            sources=empty,
        )
        assert "No subjects or sources to show." in html

    def test_labels_are_escaped(self, subjectsources, sources):
        evil = pd.DataFrame([{"id": "amani", "name": "</script><b>x</b>"}])
        html = draw_subject_source_diagram(
            subjects=evil, subjectsources=subjectsources, sources=sources
        )
        assert "</script><b>" not in html


class TestRegressions:
    def test_er_open_start_sentinel_is_not_flagged(self, subjects, sources):
        links = pd.DataFrame(
            [
                {
                    "subject": "amani",
                    "source": "s1",
                    "assigned_range": {
                        "lower": "0001-01-01T00:00:00+00:00",
                        "upper": "9999-12-31T23:59:59+00:00",
                    },
                }
            ]
        )
        _, edges = build_subject_source_graph(subjects, links, sources)
        assert edges[0]["label"] == "? → present"
        assert edges[0]["bad_date"] is False


class TestStyles:
    def test_defaults_match_original_palette(self):
        assert NodeStyle().subject_fill == "#dbeafe"
        assert NodeStyle().unlinked_border_color == "#f59e0b"
        assert LinkStyle().bad_date_color == "#dc2626"
        assert DiagramLayoutStyle().plot_bgcolor == "#ffffff"

    def test_node_colors_follow_state(self, subjects, subjectsources, sources):
        nodes, _ = build_subject_source_graph(subjects, subjectsources, sources)
        style = NodeStyle()
        assert node_colors(_node(nodes, "subject:amani"), style) == (
            style.subject_fill,
            style.border_color,
            style.border_width,
        )
        assert (
            node_colors(_node(nodes, "subject:zawadi"), style)[0]
            == style.inactive_subject_fill
        )
        assert node_colors(_node(nodes, "source:spare"), style)[1:] == (
            style.unlinked_border_color,
            style.flagged_border_width,
        )

    def test_link_colors_follow_state(self, subjects, subjectsources, sources):
        _, edges = build_subject_source_graph(subjects, subjectsources, sources)
        style = LinkStyle()
        assert link_colors(_edge(edges, "amani", "s1"), style) == (
            style.current_color,
            style.label_color,
            style.current_width,
            "",
        )
        assert link_colors(_edge(edges, "amani", "s0"), style)[3] == style.ended_dash
        assert link_colors(_edge(edges, "baraka", "s2"), style)[:2] == (
            style.bad_date_color,
            style.bad_date_color,
        )

    def test_custom_styles_are_rendered(self, subjects, subjectsources, sources):
        html = draw_subject_source_diagram(
            subjects=subjects,
            subjectsources=subjectsources,
            sources=sources,
            node_style=NodeStyle(
                subject_fill="#ff00aa", unlinked_border_color="purple"
            ),
            link_style=LinkStyle(current_color="rgb(1, 2, 3)", ended_dash=""),
            layout_style=DiagramLayoutStyle(
                font_family="Georgia, serif", subject_header="Animals", showlegend=False
            ),
        )
        assert 'fill="#ff00aa"' in html
        assert 'stroke="purple"' in html
        assert 'stroke="rgb(1, 2, 3)"' in html
        assert "stroke-dasharray" not in html
        assert "font-family: Georgia, serif" in html
        assert ">Animals</text>" in html
        assert 'class="legend"' not in html
        assert 'fill="#dbeafe"' not in html

    def test_styles_accept_spec_style_dicts(self):
        # Workflow specs pass styles as plain mappings; pydantic fills in the rest.
        style = NodeStyle.model_validate({"source_fill": "#000"})
        assert style.source_fill == "#000"
        assert style.subject_fill == "#dbeafe"

    def test_legend_uses_active_colors(self, subjects, subjectsources, sources):
        html = draw_subject_source_diagram(
            subjects=subjects,
            subjectsources=subjectsources,
            sources=sources,
            link_style=LinkStyle(bad_date_color="#123456"),
        )
        legend = html[html.index('class="legend"') : html.index('class="scroll"')]
        assert "#123456" in legend

    @pytest.mark.parametrize(
        "bad",
        [
            {"subject_fill": 'red" onload="alert(1)'},
            {"subject_fill": "red; } body { display:none"},
        ],
    )
    def test_style_values_cannot_inject_markup(self, bad):
        with pytest.raises(ValidationError):
            NodeStyle(**bad)

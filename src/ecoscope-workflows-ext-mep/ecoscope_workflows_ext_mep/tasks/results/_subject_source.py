import html
from typing import Annotated, Any

import pandas as pd
from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

# CSS colour values: hex, names, rgb()/hsl() and the like. Restricted so a style value
# can't break out of the attribute or <style> block it is written into.
COLOR_PATTERN = r"^[#\w(),.%\s-]+$"
FONT_PATTERN = r"^[\w\s,'\"-]+$"
DASH_PATTERN = r"^[\d.\s,]*$"


def _color(default: str, description: str) -> Any:
    return AdvancedField(default=default, description=description, pattern=COLOR_PATTERN)


class NodeStyle(BaseModel):
    """Style of the subject and source pills."""

    subject_fill: Annotated[str, _color("#dbeafe", "Fill colour of active subjects.")] = "#dbeafe"
    inactive_subject_fill: Annotated[str, _color("#f1f5f9", "Fill colour of inactive subjects.")] = "#f1f5f9"
    source_fill: Annotated[str, _color("#dcfce7", "Fill colour of sources.")] = "#dcfce7"
    border_color: Annotated[str, _color("#94a3b8", "Outline colour of linked subjects and sources.")] = "#94a3b8"
    unlinked_border_color: Annotated[
        str,
        _color(
            "#f59e0b",
            "Outline colour of subjects with no source and sources with no subject.",
        ),
    ] = "#f59e0b"
    missing_source_border_color: Annotated[
        str,
        _color(
            "#dc2626",
            "Outline colour of sources referenced by an assignment but missing from the sources list.",
        ),
    ] = "#dc2626"
    border_width: Annotated[
        float,
        AdvancedField(default=1.0, ge=0, description="Outline width of linked pills."),
    ] = 1.0
    flagged_border_width: Annotated[
        float,
        AdvancedField(default=2.0, ge=0, description="Outline width of unlinked or missing pills."),
    ] = 2.0
    width: Annotated[
        int,
        AdvancedField(default=210, ge=60, description="Width of each pill in pixels."),
    ] = 210
    max_label_chars: Annotated[
        int,
        AdvancedField(
            default=28,
            ge=4,
            description="Longer names are cut off with '…'; hover shows the full name.",
        ),
    ] = 28


class LinkStyle(BaseModel):
    """Style of the assignment lines and their date labels."""

    current_color: Annotated[str, _color("#475569", "Line colour of current assignments.")] = "#475569"
    ended_color: Annotated[str, _color("#cbd5e1", "Line colour of ended assignments.")] = "#cbd5e1"
    bad_date_color: Annotated[
        str,
        _color("#dc2626", "Line and label colour of assignments with an unreadable date."),
    ] = "#dc2626"
    current_width: Annotated[
        float,
        AdvancedField(default=1.5, ge=0, description="Line width of current assignments."),
    ] = 1.5
    ended_width: Annotated[
        float,
        AdvancedField(default=1.0, ge=0, description="Line width of ended assignments."),
    ] = 1.0
    ended_dash: Annotated[
        str,
        AdvancedField(
            default="4 3",
            pattern=DASH_PATTERN,
            description="Dash pattern of ended assignments as SVG stroke-dasharray (e.g. '4 3'). Empty for solid.",
        ),
    ] = "4 3"
    label_color: Annotated[str, _color("#334155", "Date label colour of current assignments.")] = "#334155"
    ended_label_color: Annotated[str, _color("#94a3b8", "Date label colour of ended assignments.")] = "#94a3b8"
    label_font_size: Annotated[
        float,
        AdvancedField(default=11, ge=6, description="Font size of the date labels in pixels."),
    ] = 11


class DiagramLayoutStyle(BaseModel):
    """Page-level style: fonts, background, column headings and spacing."""

    font_size: Annotated[float, AdvancedField(default=12, ge=6, description="Font size in pixels.")] = 12
    font_color: Annotated[str, _color("#0f172a", "Text colour.")] = "#0f172a"
    font_family: Annotated[
        str,
        AdvancedField(
            default="Helvetica, Arial, sans-serif",
            pattern=FONT_PATTERN,
            description="CSS font family.",
        ),
    ] = "Helvetica, Arial, sans-serif"
    plot_bgcolor: Annotated[str, _color("#ffffff", "Background colour.")] = "#ffffff"
    showlegend: Annotated[
        bool,
        AdvancedField(default=True, description="Show the legend above the diagram."),
    ] = True
    muted_color: Annotated[str, _color("#64748b", "Colour of the legend and empty-state text.")] = "#64748b"
    header_color: Annotated[str, _color("#475569", "Colour of the column headings.")] = "#475569"
    section_color: Annotated[str, _color("#b45309", "Colour of the 'no source' / 'no subject' headings.")] = "#b45309"
    rule_color: Annotated[str, _color("#e2e8f0", "Colour of the line above the unlinked section.")] = "#e2e8f0"
    subject_header: Annotated[str, AdvancedField(default="Subjects", description="Left column heading.")] = "Subjects"
    assigned_header: Annotated[str, AdvancedField(default="Assigned", description="Date column heading.")] = "Assigned"
    source_header: Annotated[str, AdvancedField(default="Sources", description="Right column heading.")] = "Sources"
    label_column_width: Annotated[
        int,
        AdvancedField(default=150, ge=40, description="Width of the date label column in pixels."),
    ] = 150
    link_width: Annotated[
        int,
        AdvancedField(
            default=130,
            ge=20,
            description="Horizontal space for the lines into the sources column.",
        ),
    ] = 130
    hover_dim_opacity: Annotated[
        float,
        AdvancedField(
            default=0.12,
            ge=0.0,
            le=1.0,
            description="Opacity of unrelated items while hovering a subject or source.",
        ),
    ] = 0.12


# ER stores open-ended assignments with an upper bound in year 9999, and a missing
# start as year 0001 (Python's datetime.min).
OPEN_ENDED_YEAR = 9999
OPEN_START_YEAR = 1


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # list/dict values
        return False


def _first(row: pd.Series, *columns: str) -> Any:
    for col in columns:
        value = row.get(col)
        if not _is_missing(value):
            return value
    return None


def _is_inactive(row: pd.Series) -> bool:
    value = row.get("is_active")
    return not _is_missing(value) and not bool(value)


def _parse_bound(value: Any, invalid: Any) -> tuple[pd.Timestamp | None, str | None]:
    """Return (timestamp, bad_raw) for one end of an assignment range.

    The timestamp is None for an open end (missing or an ER sentinel). `bad_raw` is
    the first 10 characters of a date that could not be read, else None.
    """
    if invalid:
        return None, str(invalid)[:10]
    if value is None:
        return None, None
    # Checked before parsing: both sentinels are outside pandas' Timestamp range and would coerce to NaT.
    if str(value)[:4] in (f"{OPEN_ENDED_YEAR:04d}", f"{OPEN_START_YEAR:04d}"):
        return None, None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None, str(value)[:10]
    if ts.year >= OPEN_ENDED_YEAR:
        return None, None
    return ts, None


def assignment_bounds(
    row: pd.Series,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str | None, str | None]:
    """Return (start, end, bad_start, bad_end) for a subjectsource row.

    `start`/`end` are None when that end is open; `bad_start`/`bad_end` hold the raw
    text of a date that could not be read. Accepts either the raw ER shape
    (`assigned_range` dict) or the flattened `assigned_range_lower`/`assigned_range_upper`
    columns, with optional `invalid_assigned_range_*` columns carrying unparsable dates.
    """
    raw = row.get("assigned_range")
    raw = raw if isinstance(raw, dict) else {}
    lower = _first(row, "assigned_range_lower")
    lower = raw.get("lower") if lower is None else lower
    upper = _first(row, "assigned_range_upper")
    upper = raw.get("upper") if upper is None else upper

    start, bad_start = _parse_bound(lower, _first(row, "invalid_assigned_range_lower"))
    end, bad_end = _parse_bound(upper, _first(row, "invalid_assigned_range_upper"))
    return start, end, bad_start, bad_end


def format_assignment_range(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    bad_start: str | None,
    bad_end: str | None,
) -> str:
    """Render bounds from `assignment_bounds` as e.g. '2024-03-01 → present'."""
    start_label = f"⚠ {bad_start}" if bad_start else start.strftime("%Y-%m-%d") if start is not None else "?"
    end_label = f"⚠ {bad_end}" if bad_end else end.strftime("%Y-%m-%d") if end is not None else "present"
    return f"{start_label} → {end_label}"


def _assignment_range(row: pd.Series) -> tuple[str, bool, bool]:
    """Return (label, is_current, has_bad_date) for a subjectsource row."""
    start, end, bad_start, bad_end = assignment_bounds(row)
    is_current = end is None and bad_end is None
    return (
        format_assignment_range(start, end, bad_start, bad_end),
        is_current,
        bool(bad_start or bad_end),
    )


def _source_label(row: pd.Series) -> str:
    return str(_first(row, "manufacturer_id", "collar_id", "id"))


def _source_tooltip(row: pd.Series) -> str:
    parts = [_source_label(row)]
    for col in ("provider", "source_type", "model_name"):
        value = _first(row, col)
        if value is not None:
            parts.append(f"{col.replace('_', ' ')}: {value}")
    return "\n".join(parts)


def _subject_tooltip(row: pd.Series) -> str:
    parts = [str(_first(row, "name", "id"))]
    for col in ("subject_type", "subject_subtype"):
        value = _first(row, col)
        if value is not None:
            parts.append(f"{col.replace('_', ' ')}: {value}")
    if _is_inactive(row):
        parts.append("inactive")
    return "\n".join(parts)


def build_subject_source_graph(
    subjects: pd.DataFrame,
    subjectsources: pd.DataFrame,
    sources: pd.DataFrame,
    include_unlinked_sources: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Build node and edge dicts for subjects, sources and their assignments.

    Nodes and edges carry state flags only (`linked`, `inactive`, `missing`,
    `current`, `bad_date`); colours are applied at render time from the style models.
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    subject_ids = {str(i) for i in subjects["id"]} if "id" in subjects else set()
    links = subjectsources.copy() if not subjectsources.empty else pd.DataFrame(columns=["subject", "source"])
    links["subject"] = links["subject"].astype(str)
    links["source"] = links["source"].astype(str)
    links = links[links["subject"].isin(subject_ids)]
    linked_source_ids = set(links["source"])
    linked_subject_ids = set(links["subject"])

    for _, row in subjects.iterrows():
        sid = str(row["id"])
        nodes.append(
            {
                "id": f"subject:{sid}",
                "label": str(_first(row, "name", "id")),
                "title": _subject_tooltip(row),
                "group": "subject",
                "linked": sid in linked_subject_ids,
                "inactive": _is_inactive(row),
                "missing": False,
            }
        )

    known_source_ids: set[str] = set()
    for _, row in sources.iterrows():
        src_id = str(row["id"])
        linked = src_id in linked_source_ids
        if not linked and not include_unlinked_sources:
            continue
        known_source_ids.add(src_id)
        nodes.append(
            {
                "id": f"source:{src_id}",
                "label": _source_label(row),
                "title": _source_tooltip(row),
                "group": "source",
                "linked": linked,
                "inactive": False,
                "missing": False,
            }
        )

    for _, row in links.iterrows():
        src_id = row["source"]
        if src_id not in known_source_ids:
            # Assignment points at a source missing from the sources list; still draw it.
            known_source_ids.add(src_id)
            nodes.append(
                {
                    "id": f"source:{src_id}",
                    "label": src_id[:8],
                    "title": f"{src_id}\n(not found in sources)",
                    "group": "source",
                    "linked": True,
                    "inactive": False,
                    "missing": True,
                }
            )
        label, is_current, bad_date = _assignment_range(row)
        edges.append(
            {
                "from": f"subject:{row['subject']}",
                "to": f"source:{src_id}",
                "label": label,
                "title": label,
                "current": is_current,
                "bad_date": bad_date,
            }
        )

    return nodes, edges


def layout_subject_source_diagram(nodes: list[dict], edges: list[dict]) -> dict[str, Any]:
    """Assign each subject, source and assignment a row in a two-column layout.

    Linked subjects are listed alphabetically on the left, each spanning one row per
    assignment. Each linked source is placed as close as possible to the rows of the
    assignments pointing at it, which keeps crossing lines to a minimum. Subjects with
    no source and sources with no subject are listed at the bottom, under their own
    headings.

    Returns:
        A dict with `subjects` and `sources` (node dicts plus a `row`), `links` (edge
        dicts plus a `row`), `unlinked_row` (the row of the unlinked section heading,
        or None) and `n_rows`.
    """
    by_subject: dict[str, list[dict]] = {}
    for edge in edges:
        by_subject.setdefault(edge["from"], []).append(edge)

    def name_key(node: dict) -> str:
        return node["label"].casefold()

    subjects = sorted((n for n in nodes if n["group"] == "subject"), key=name_key)
    sources = sorted((n for n in nodes if n["group"] == "source"), key=name_key)
    linked_subjects = [n for n in subjects if n["id"] in by_subject]
    linked_source_ids = {e["to"] for e in edges}

    placed_subjects: list[dict] = []
    links: list[dict] = []
    row = 0
    for node in linked_subjects:
        own = sorted(by_subject[node["id"]], key=lambda e: e["label"])
        placed_subjects.append({**node, "row": row + (len(own) - 1) / 2})
        for edge in own:
            links.append({**edge, "row": row})
            row += 1

    # Each source wants to sit at the mean row of its assignments; walk them in that
    # order and push down whenever one would overlap the previous.
    rows_by_source: dict[str, list[int]] = {}
    for link in links:
        rows_by_source.setdefault(link["to"], []).append(link["row"])
    wanted = sorted(
        (
            (sum(rows_by_source[n["id"]]) / len(rows_by_source[n["id"]]), n)
            for n in sources
            if n["id"] in linked_source_ids
        ),
        key=lambda pair: pair[0],
    )
    placed_sources: list[dict] = []
    next_free = 0.0
    for desired, node in wanted:
        source_row = max(desired, next_free)
        placed_sources.append({**node, "row": source_row})
        next_free = source_row + 1

    unlinked_subjects = [n for n in subjects if n["id"] not in by_subject]
    unlinked_sources = [n for n in sources if n["id"] not in linked_source_ids]
    end = max(row, int(next_free + 0.999))
    unlinked_row = None
    if unlinked_subjects or unlinked_sources:
        unlinked_row = end + (1 if end else 0)
        for i, node in enumerate(unlinked_subjects):
            placed_subjects.append({**node, "row": unlinked_row + 1 + i})
        for i, node in enumerate(unlinked_sources):
            placed_sources.append({**node, "row": unlinked_row + 1 + i})
        end = unlinked_row + 1 + max(len(unlinked_subjects), len(unlinked_sources))

    return {
        "subjects": placed_subjects,
        "sources": placed_sources,
        "links": links,
        "unlinked_row": unlinked_row,
        "n_rows": end,
    }


# Fixed spacing, in pixels. Widths that are worth changing live on the style models.
PAD = 16
HEADER_H = 34
LABEL_GAP = 14


def node_colors(node: dict, style: NodeStyle) -> tuple[str, str, float]:
    """Return (fill, border, border_width) for a node."""
    if node["group"] == "subject":
        fill = style.inactive_subject_fill if node["inactive"] else style.subject_fill
    else:
        fill = style.source_fill
    if node["missing"]:
        return fill, style.missing_source_border_color, style.flagged_border_width
    if not node["linked"]:
        return fill, style.unlinked_border_color, style.flagged_border_width
    return fill, style.border_color, style.border_width


def link_colors(link: dict, style: LinkStyle) -> tuple[str, str, float, str]:
    """Return (line colour, label colour, width, dash) for a link."""
    width = style.current_width if link["current"] else style.ended_width
    dash = "" if link["current"] else style.ended_dash
    if link["bad_date"]:
        return style.bad_date_color, style.bad_date_color, width, dash
    if link["current"]:
        return style.current_color, style.label_color, width, dash
    return style.ended_color, style.ended_label_color, width, dash


def _legend(node_style: NodeStyle, link_style: LinkStyle) -> str:
    """Legend built from the active styles, so it always matches the diagram."""
    esc = html.escape

    def line(color: str, width: float, dash: str, text: str) -> str:
        dash_attr = f' stroke-dasharray="{esc(dash)}"' if dash else ""
        return (
            f'<span class="key"><svg width="26" height="10"><line x1="1" x2="25" y1="5" y2="5" '
            f'stroke="{esc(color)}" stroke-width="{width}"{dash_attr}/></svg>{text}</span>'
        )

    def swatch(fill: str, border: str, width: float, text: str) -> str:
        return (
            f'<span class="key"><svg width="26" height="14"><rect x="1" y="1" width="24" height="12" rx="6" '
            f'fill="{esc(fill)}" stroke="{esc(border)}" stroke-width="{width}"/></svg>{text}</span>'
        )

    ns, ls = node_style, link_style
    return (
        '<div class="legend">'
        + line(ls.current_color, ls.current_width, "", "current")
        + line(ls.ended_color, ls.ended_width, ls.ended_dash, "ended")
        + line(ls.bad_date_color, ls.current_width, "", "bad date")
        + swatch(
            ns.subject_fill,
            ns.unlinked_border_color,
            ns.flagged_border_width,
            "no link",
        )
        + swatch(
            ns.inactive_subject_fill,
            ns.border_color,
            ns.border_width,
            "inactive subject",
        )
        + swatch(
            ns.source_fill,
            ns.missing_source_border_color,
            ns.flagged_border_width,
            "source not found",
        )
        + '<span class="hint">Hover a pill to trace its links.</span></div>'
    )


def _render_html(
    layout: dict[str, Any],
    title: str,
    row_height: int,
    div_id: str,
    node_style: NodeStyle,
    link_style: LinkStyle,
    layout_style: DiagramLayoutStyle,
) -> str:
    esc = html.escape
    ns, ls, ly = node_style, link_style, layout_style
    pill_h = max(16, min(30, row_height - 10))
    subject_x = PAD
    label_x = subject_x + ns.width + LABEL_GAP
    source_x = label_x + ly.label_column_width + ly.link_width
    diagram_w = source_x + ns.width + PAD

    def y_of(row: float) -> float:
        return HEADER_H + row * row_height + row_height / 2

    def truncate(text: str) -> str:
        n = ns.max_label_chars
        return text if len(text) <= n else text[: n - 1] + "…"

    subject_idx = {n["id"]: i for i, n in enumerate(layout["subjects"])}
    source_idx = {n["id"]: i for i, n in enumerate(layout["sources"])}
    subject_y = {n["id"]: y_of(n["row"]) for n in layout["subjects"]}
    source_y = {n["id"]: y_of(n["row"]) for n in layout["sources"]}

    # Every element carries the s-<i>/c-<j> classes of what it is connected to, so
    # hovering a subject (or source) can highlight everything sharing its class.
    subject_classes: dict[str, set[str]] = {n["id"]: {f"s-{subject_idx[n['id']]}"} for n in layout["subjects"]}
    source_classes: dict[str, set[str]] = {n["id"]: {f"c-{source_idx[n['id']]}"} for n in layout["sources"]}
    for link in layout["links"]:
        subject_classes[link["from"]].add(f"c-{source_idx[link['to']]}")
        source_classes[link["to"]].add(f"s-{subject_idx[link['from']]}")

    parts: list[str] = []
    for link in layout["links"]:
        y0, y1, y2 = subject_y[link["from"]], y_of(link["row"]), source_y[link["to"]]
        x0, x1, x2 = subject_x + ns.width, label_x, source_x
        lx = label_x + ly.label_column_width
        stroke, text_fill, width, dash = link_colors(link, ls)
        stroke_attrs = f'stroke="{esc(stroke)}" stroke-width="{width}" fill="none"' + (
            f' stroke-dasharray="{esc(dash)}"' if dash else ""
        )
        classes = f"el s-{subject_idx[link['from']]} c-{source_idx[link['to']]}"
        parts.append(
            f'<g class="{classes}"><title>{esc(link["title"])}</title>'
            f'<path d="M{x0},{y0} C{x0 + LABEL_GAP / 2},{y0} {x1 - LABEL_GAP / 2},{y1} {x1},{y1}" {stroke_attrs}/>'
            f'<text x="{x1 + 4}" y="{y1}" class="date" fill="{esc(text_fill)}">{esc(link["label"])}</text>'
            f'<path d="M{lx},{y1} C{lx + ly.link_width / 2},{y1} {x2 - ly.link_width / 2},{y2} {x2},{y2}" '
            f"{stroke_attrs}/></g>"
        )

    def pill(node: dict, x: float, y: float, classes: set[str], key: str) -> str:
        fill, border, border_w = node_colors(node, ns)
        return (
            f'<g class="el node {" ".join(sorted(classes))}" data-k="{key}">'
            f"<title>{esc(node['title'])}</title>"
            f'<rect x="{x}" y="{y - pill_h / 2}" width="{ns.width}" height="{pill_h}" rx="{pill_h / 2}" '
            f'fill="{esc(fill)}" stroke="{esc(border)}" stroke-width="{border_w}"/>'
            f'<text x="{x + 12}" y="{y}">{esc(truncate(node["label"]))}</text></g>'
        )

    for node in layout["subjects"]:
        key = f"s-{subject_idx[node['id']]}"
        parts.append(pill(node, subject_x, subject_y[node["id"]], subject_classes[node["id"]], key))
    for node in layout["sources"]:
        key = f"c-{source_idx[node['id']]}"
        parts.append(pill(node, source_x, source_y[node["id"]], source_classes[node["id"]], key))

    if layout["unlinked_row"] is not None:
        y = y_of(layout["unlinked_row"])
        rule_y = y - row_height / 2
        parts.append(
            f'<line x1="{PAD}" x2="{diagram_w - PAD}" y1="{rule_y}" y2="{rule_y}" class="rule"/>'
            f'<text x="{subject_x}" y="{y}" class="section">Subjects with no source</text>'
            f'<text x="{source_x}" y="{y}" class="section">Sources with no subject</text>'
        )

    height = HEADER_H + layout["n_rows"] * row_height + PAD
    heading = f"<h3>{esc(title)}</h3>" if title else ""
    if not layout["subjects"] and not layout["sources"]:
        body = '<p class="empty">No subjects or sources to show.</p>'
    else:
        legend = _legend(ns, ls) if ly.showlegend else ""
        body = (
            f'{legend}<div class="scroll"><svg id="{div_id}" width="{diagram_w}" height="{height}" '
            f'viewBox="0 0 {diagram_w} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{subject_x}" y="{HEADER_H / 2}" class="col">{esc(ly.subject_header)}</text>'
            f'<text x="{label_x}" y="{HEADER_H / 2}" class="col">{esc(ly.assigned_header)}</text>'
            f'<text x="{source_x}" y="{HEADER_H / 2}" class="col">{esc(ly.source_header)}</text>'
            f"{''.join(parts)}</svg></div>"
        )
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ margin: 0; background: {ly.plot_bgcolor}; font-family: {ly.font_family}; color: {ly.font_color}; }}
  h3 {{ margin: 8px {PAD}px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: center;
             margin: 8px {PAD}px 0; font-size: {ly.font_size}px; color: {ly.muted_color}; }}
  .legend .key {{ display: inline-flex; align-items: center; gap: 5px; }}
  .empty {{ margin: {PAD}px; color: {ly.muted_color}; }}
  .scroll {{ overflow-x: auto; }}
  svg text {{ font-size: {ly.font_size}px; dominant-baseline: central; fill: {ly.font_color}; }}
  svg text.date {{ font-size: {ls.label_font_size}px; font-variant-numeric: tabular-nums; }}
  svg text.col {{ font-weight: 600; fill: {ly.header_color}; text-transform: uppercase; letter-spacing: .04em; }}
  svg text.section {{ font-weight: 600; fill: {ly.section_color}; }}
  svg line.rule {{ stroke: {ly.rule_color}; }}
  svg .node {{ cursor: pointer; }}
  svg.dim .el {{ opacity: {ly.hover_dim_opacity}; }}
  svg.dim .el.on {{ opacity: 1; }}
</style>
</head>
<body>
{heading}{body}
<script>
  const svg = document.getElementById("{div_id}");
  if (svg) {{
    svg.querySelectorAll("[data-k]").forEach((node) => {{
      const key = node.dataset.k;
      node.addEventListener("mouseenter", () => {{
        svg.classList.add("dim");
        svg.querySelectorAll("." + key).forEach((el) => el.classList.add("on"));
      }});
      node.addEventListener("mouseleave", () => {{
        svg.classList.remove("dim");
        svg.querySelectorAll(".on").forEach((el) => el.classList.remove("on"));
      }});
    }});
  }}
</script>
</body>
</html>"""


@register()
def draw_subject_source_diagram(
    subjects: AnyDataFrame,
    subjectsources: AnyDataFrame,
    sources: AnyDataFrame,
    title: Annotated[str, Field(description="Heading shown above the diagram. Empty for none.")] = "",
    row_height: Annotated[
        int,
        Field(
            ge=16,
            description=(
                "Height in pixels of each row; every assignment, and every unlinked subject or source, gets a row."
            ),
        ),
    ] = 42,
    include_unlinked_sources: Annotated[
        bool,
        Field(description="Also draw sources that are not assigned to any of the given subjects."),
    ] = True,
    node_style: Annotated[
        NodeStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Colours, outlines and size of the subject and source pills.",
        ),
    ] = None,
    link_style: Annotated[
        LinkStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Colours, widths and dash pattern of the assignment lines.",
        ),
    ] = None,
    layout_style: Annotated[
        DiagramLayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Fonts, background, legend, column headings and spacing.",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description=(
                "The id of the dashboard widget that this tile layer belongs to. "
                "If set this MUST match the widget title as defined downstream in create_widget tasks"
            ),
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    """Draw subjects and sources in two columns, linked by their assignments.

    Subjects are listed on the left and sources on the right. Each assignment is a
    line between them, labelled with its date range. Current assignments are solid,
    ended ones are dashed, and lines with unreadable dates are flagged. Subjects with
    no source and sources with no subject are listed at the bottom with a highlighted
    outline. Hovering a subject or source highlights everything it is linked to.

    All colours, widths and fonts come from `node_style`, `link_style` and
    `layout_style`; any field left unset keeps its default.

    Returns:
        A self-contained HTML document with the diagram as inline SVG.
    """
    nodes, edges = build_subject_source_graph(
        subjects=subjects,
        subjectsources=subjectsources,
        sources=sources,
        include_unlinked_sources=include_unlinked_sources,
    )
    layout = layout_subject_source_diagram(nodes, edges)
    div_id = "".join(c if c.isalnum() else "-" for c in (widget_id or "subject-source-diagram"))
    return _render_html(
        layout,
        title=title,
        row_height=row_height,
        div_id=div_id,
        node_style=node_style or NodeStyle(),
        link_style=link_style or LinkStyle(),
        layout_style=layout_style or DiagramLayoutStyle(),
    )

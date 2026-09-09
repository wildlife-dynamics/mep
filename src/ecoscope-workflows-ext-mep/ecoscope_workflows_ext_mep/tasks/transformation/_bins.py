from typing import Literal
import numpy as np
import pandas as pd
from matplotlib import colormaps
from wt_registry import register
from matplotlib.colors import to_hex
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope_workflows_ext_custom.tasks.transformation._color_utils import ColorPalette, CustomPalette


_MC_SCHEME = {
    "equal_interval": "equalinterval",
    "quantile": "quantiles",
    "natural_breaks": "naturalbreaks",
    "fisher_jenks": "fisherjenks",
    "std_mean": "stdmean",
}

def _compute_edges(vals: pd.Series, scheme: str, bins: int) -> np.ndarray:
    """Bin edges from a mapclassify scheme, clipped to [vals.min(), vals.max()]."""
    import mapclassify

    name = _MC_SCHEME.get(scheme)
    if name is None:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose from {list(_MC_SCHEME)}.")

    lo, hi = float(vals.min()), float(vals.max())
    y = vals.to_numpy()
    # std_mean is defined by std multiples, not a class count, so k doesn't apply
    mc = mapclassify.classify(y, name) if name == "stdmean" else mapclassify.classify(y, name, k=bins)

    # keep only breaks strictly inside the data range (drops std_mean's out-of-range edges)
    interior = [b for b in np.asarray(mc.bins, dtype=float) if lo < b < hi]
    return np.unique(np.array([lo, *interior, hi], dtype=float))

@register()
def add_visit_bins(
    df: AnyDataFrame,
    col: str,
    mask_col: str,
    new_col: str,
    bins: int = 5,
    scheme: Literal["equal_interval", "quantile", "natural_breaks", "fisher_jenks", "std_mean"] = "natural_breaks",
    no_data_label: str = "Unvisited",
    use_abs: bool = True,
) -> AnyDataFrame:
    """Classify a numeric column into labeled bins, for rows where `mask_col` is true.

    Rows excluded by `mask_col` (or with a null `col`) get `no_data_label`
    instead of a bin label.
    """
    # included AND has a real value
    mask = df[mask_col].fillna(False) & df[col].notna()
    vals = df.loc[mask, col]
    if use_abs:
        vals = vals.abs()
    df[new_col] = no_data_label
    def fmt(x):
        return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")

    if vals.empty:
        labels = []
    elif vals.nunique() == 1:
        labels = [fmt(vals.iloc[0])]
        df.loc[mask, new_col] = labels[0]
    else:
        edges = np.unique(np.round(_compute_edges(vals, scheme, bins), 2))
        # don't let rounding pull the outer edges inward and orphan min/max
        edges[0] = min(edges[0], vals.min())
        edges[-1] = max(edges[-1], vals.max())

        if len(edges) < 2:  # everything collapsed to one edge
            labels = [fmt(vals.iloc[0])]
            df.loc[mask, new_col] = labels[0]
        else:
            labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(len(edges) - 1)]
            binned = pd.cut(vals, bins=edges, labels=labels, include_lowest=True)
            df.loc[mask, new_col] = binned.astype("object").where(binned.notna(), no_data_label)

    df[new_col] = pd.Categorical(df[new_col], categories=[no_data_label] + labels, ordered=True)
    return df

@register()
def add_bin_colors(
    df: AnyDataFrame, 
    col: str, 
    new_col: str, 
    cmap: ColorPalette, 
    no_data_label: str = "Unvisited",
    no_data_color: str = "#808080",
    ):
    df = df.copy()
    # use the categorical's own order if present, else sorted uniques
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        cats = list(df[col].cat.categories)
    else:
        cats = sorted(df[col].dropna().unique())

    ramp_cats = [c for c in cats if c != no_data_label]

    if isinstance(cmap, CustomPalette):
        colors = cmap.colors
        color_map = {c: colors[i % len(colors)] for i, c in enumerate(ramp_cats)}
    else:
        cm = colormaps[cmap.name]
        # evenly spaced samples across the colormap, one per non-no-data bin
        points = np.linspace(0, 1, len(ramp_cats)) if len(ramp_cats) > 1 else [0.5]
        color_map = {c: to_hex(cm(p)) for c, p in zip(ramp_cats, points)}
    color_map[no_data_label] = no_data_color

    df[new_col] = df[col].map(color_map)
    return df
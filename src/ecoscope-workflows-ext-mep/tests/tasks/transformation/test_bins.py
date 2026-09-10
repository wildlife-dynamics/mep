"""Tests for ecoscope_workflows_ext_mep.tasks.transformation._bins.

Every function here is registered via `wt_registry.register()`, a no-op
decorator at call time, so each is called directly as plain Python.

`add_bin_colors`'s `cmap` parameter is typed as `ColorPalette`, which is
a discriminated `Union[ColormapPalette, CustomPalette]` type alias, not
a class -- the concrete variants (`ColormapPalette`, `CustomPalette`) are
what actually get constructed and passed in.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_ste.tasks.results._spatial_layers import (
    ColormapPalette,
    CustomPalette,
)
from ecoscope_workflows_ext_mep.tasks.transformation._bins import (
    add_bin_colors,
    add_visit_bins,
    order_bin_categories,
)


class TestAddVisitBins:
    def test_rows_excluded_by_mask_get_the_no_data_label(self):
        df = pd.DataFrame({"val": [1, 5], "mask": [False, True]})

        result = add_visit_bins(df, col="val", mask_col="mask", new_col="bin", no_data_label="Unvisited")

        assert result["bin"].tolist() == ["Unvisited", "5"]

    def test_null_value_gets_no_data_label_even_when_mask_is_true(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0], "mask": [True, True, True]})

        result = add_visit_bins(df, col="val", mask_col="mask", new_col="bin", no_data_label="Unvisited")

        assert result.loc[1, "bin"] == "Unvisited"

    def test_a_null_mask_value_is_treated_as_excluded(self):
        df = pd.DataFrame({"val": [5.0], "mask": [None]})

        result = add_visit_bins(df, col="val", mask_col="mask", new_col="bin", no_data_label="Unvisited")

        assert result["bin"].tolist() == ["Unvisited"]

    def test_single_unique_included_value_becomes_its_own_label_not_a_range(self):
        df = pd.DataFrame({"val": [5, 5, 5], "mask": [True, True, True]})

        result = add_visit_bins(df, col="val", mask_col="mask", new_col="bin")

        assert result["bin"].tolist() == ["5", "5", "5"]

    def test_use_abs_applies_before_binning(self):
        df = pd.DataFrame({"val": [-5, -5], "mask": [True, True]})

        result = add_visit_bins(df, col="val", mask_col="mask", new_col="bin", use_abs=True)

        assert result["bin"].tolist() == ["5", "5"]

    def test_use_abs_false_keeps_the_sign(self):
        df = pd.DataFrame({"val": [-5, -5], "mask": [True, True]})

        result = add_visit_bins(df, col="val", mask_col="mask", new_col="bin", use_abs=False)

        assert result["bin"].tolist() == ["-5", "-5"]

    def test_no_included_rows_leaves_only_the_no_data_category(self):
        df = pd.DataFrame({"val": [1, 2, 3], "mask": [False, False, False]})

        result = add_visit_bins(df, col="val", mask_col="mask", new_col="bin", no_data_label="Unvisited")

        assert result["bin"].tolist() == ["Unvisited", "Unvisited", "Unvisited"]
        assert result["bin"].cat.categories.tolist() == ["Unvisited"]

    def test_multiple_values_are_split_into_labeled_equal_interval_ranges(self):
        # equal_interval with a known min/max and bin count gives deterministic
        # edges, unlike natural_breaks/quantile/etc which depend on mapclassify
        # internals -- this pins the labeling/formatting behavior instead.
        df = pd.DataFrame({"val": [0, 2, 4, 6, 8, 10], "mask": [True] * 6})

        result = add_visit_bins(
            df, col="val", mask_col="mask", new_col="bin", bins=5, scheme="equal_interval", no_data_label="Unvisited"
        )

        assert result["bin"].tolist() == ["0–2", "0–2", "2–4", "4–6", "6–8", "8–10"]
        assert result["bin"].cat.categories.tolist() == ["Unvisited", "0–2", "2–4", "4–6", "6–8", "8–10"]

    def test_result_column_is_an_ordered_categorical(self):
        df = pd.DataFrame({"val": [5], "mask": [True]})

        result = add_visit_bins(df, col="val", mask_col="mask", new_col="bin")

        assert isinstance(result["bin"].dtype, pd.CategoricalDtype)
        assert result["bin"].cat.ordered is True

    def test_unknown_scheme_raises_value_error(self):
        df = pd.DataFrame({"val": [1, 2, 3], "mask": [True, True, True]})

        with pytest.raises(ValueError, match="Unknown scheme"):
            add_visit_bins(df, col="val", mask_col="mask", new_col="bin", scheme="not-a-real-scheme")  # type: ignore[arg-type]


class TestAddBinColors:
    def _categorical_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"bin": pd.Categorical(["Unvisited", "A", "B"], categories=["Unvisited", "A", "B"], ordered=True)}
        )

    def test_custom_palette_assigns_colors_by_category_order(self):
        df = self._categorical_df()

        result = add_bin_colors(
            df,
            col="bin",
            new_col="hex",
            cmap=CustomPalette(colors=["#111111", "#222222"]),
            no_data_label="Unvisited",
            no_data_color="#808080",
        )

        assert result.set_index("bin")["hex"].to_dict() == {
            "Unvisited": "#808080",
            "A": "#111111",
            "B": "#222222",
        }

    def test_custom_palette_cycles_when_there_are_more_categories_than_colors(self):
        df = pd.DataFrame(
            {"bin": pd.Categorical(["Unvisited", "A", "B", "C"], categories=["Unvisited", "A", "B", "C"], ordered=True)}
        )

        result = add_bin_colors(
            df,
            col="bin",
            new_col="hex",
            cmap=CustomPalette(colors=["#111111", "#222222"]),
            no_data_label="Unvisited",
        )

        by_bin = result.set_index("bin")["hex"]
        assert by_bin["A"] == "#111111"
        assert by_bin["B"] == "#222222"
        assert by_bin["C"] == "#111111"  # wraps back around

    def test_colormap_palette_assigns_one_color_per_non_no_data_category(self):
        df = self._categorical_df()

        result = add_bin_colors(
            df,
            col="bin",
            new_col="hex",
            cmap=ColormapPalette(name="RdYlGn"),
            no_data_label="Unvisited",
            no_data_color="#808080",
        )

        by_bin = result.set_index("bin")["hex"]
        assert by_bin["Unvisited"] == "#808080"
        # Two distinct, real hex colors sampled from the colormap.
        assert by_bin["A"] != by_bin["B"]
        assert by_bin["A"].startswith("#") and len(by_bin["A"]) == 7

    def test_no_data_label_always_gets_no_data_color_regardless_of_its_position(self):
        # order_bin_categories always sorts "Unvisited" first, but this
        # function shouldn't depend on that ordering to find it.
        df = pd.DataFrame({"bin": pd.Categorical(["A", "Unvisited"], categories=["A", "Unvisited"], ordered=True)})

        result = add_bin_colors(
            df,
            col="bin",
            new_col="hex",
            cmap=CustomPalette(colors=["#111111"]),
            no_data_label="Unvisited",
            no_data_color="#808080",
        )

        assert result.set_index("bin")["hex"]["Unvisited"] == "#808080"

    def test_non_categorical_input_column_is_sorted_before_assigning_colors(self):
        df = pd.DataFrame({"bin": ["B", "Unvisited", "A"]})

        result = add_bin_colors(
            df,
            col="bin",
            new_col="hex",
            cmap=CustomPalette(colors=["#111111", "#222222"]),
            no_data_label="Unvisited",
        )

        by_bin = result.set_index("bin")["hex"]
        # sorted(["A", "B"]) -> A gets the first color, B the second
        assert by_bin["A"] == "#111111"
        assert by_bin["B"] == "#222222"

    def test_does_not_mutate_the_input_dataframe(self):
        df = self._categorical_df()

        add_bin_colors(df, col="bin", new_col="hex", cmap=CustomPalette(colors=["#111111", "#222222"]))

        assert "hex" not in df.columns


class TestOrderBinCategories:
    def test_sorts_labels_by_their_leading_number(self):
        df = pd.DataFrame({"visit_bin": ["5-10", "Unvisited", "0-5", "10-15"]})

        result = order_bin_categories(df, bin_column="visit_bin")

        assert result["visit_bin"].cat.categories.tolist() == ["Unvisited", "0-5", "5-10", "10-15"]

    def test_labels_without_a_leading_number_sort_first(self):
        df = pd.DataFrame({"visit_bin": ["3-4", "N/A", "1-2"]})

        result = order_bin_categories(df, bin_column="visit_bin")

        assert result["visit_bin"].cat.categories.tolist()[0] == "N/A"

    def test_result_is_an_ordered_categorical(self):
        df = pd.DataFrame({"visit_bin": ["1-2", "3-4"]})

        result = order_bin_categories(df, bin_column="visit_bin")

        assert isinstance(result["visit_bin"].dtype, pd.CategoricalDtype)
        assert result["visit_bin"].cat.ordered is True

    def test_defaults_to_the_visit_bin_column(self):
        df = pd.DataFrame({"visit_bin": ["2-3", "0-1"]})

        result = order_bin_categories(df)

        assert result["visit_bin"].cat.categories.tolist() == ["0-1", "2-3"]

    def test_does_not_mutate_the_input_dataframe(self):
        df = pd.DataFrame({"visit_bin": ["1-2", "0-1"]})

        order_bin_categories(df, bin_column="visit_bin")

        assert not isinstance(df["visit_bin"].dtype, pd.CategoricalDtype)

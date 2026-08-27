"""Tests for ecoscope_workflows_ext_mep.tasks.transformation._tabular.

`dataframe_column_unique` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecoscope_workflows_ext_mep.tasks.transformation._tabular import (
    dataframe_column_unique,
)


class TestDataframeColumnUnique:
    def test_returns_unique_values_preserving_first_occurrence_order(self):
        df = pd.DataFrame({"subject_id": ["a", "b", "a", "c", "b"]})

        result = dataframe_column_unique(df=df, column_name="subject_id")

        assert list(result) == ["a", "b", "c"]

    def test_return_value_is_a_numpy_ndarray_not_a_list(self):
        # The return annotation says `list`, but `Series.unique()` actually
        # returns a numpy ndarray -- documenting the real runtime behavior
        # rather than the (inaccurate) type hint.
        df = pd.DataFrame({"x": [1, 2, 2, 3]})

        result = dataframe_column_unique(df=df, column_name="x")

        assert isinstance(result, np.ndarray)

    def test_empty_dataframe_returns_empty_array(self):
        df = pd.DataFrame({"x": []})

        result = dataframe_column_unique(df=df, column_name="x")

        assert len(result) == 0

    def test_missing_column_raises_key_error(self):
        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(KeyError):
            dataframe_column_unique(df=df, column_name="does_not_exist")

    def test_nan_is_included_as_a_unique_value(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 1.0, np.nan]})

        result = dataframe_column_unique(df=df, column_name="x")

        assert len(result) == 2
        assert 1.0 in result
        assert any(pd.isna(v) for v in result)

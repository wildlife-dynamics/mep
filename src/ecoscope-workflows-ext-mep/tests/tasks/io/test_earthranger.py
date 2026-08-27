"""Tests for ecoscope_workflows_ext_mep.tasks.io._earthranger.

`get_subjects` is registered via `wt_registry.register()`, a no-op
decorator at call time, so it is called directly as plain Python. The
`client.get_subjects(...)` EarthRanger call is faked with a small stub
object rather than a deep mock, so we can assert on exactly what
`get_subjects` passed through.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecoscope_workflows_ext_mep.tasks.io._earthranger import get_subjects


class _FakeClient:
    """Records the kwargs it was called with and returns a canned frame."""

    def __init__(self, return_value: pd.DataFrame):
        self.return_value = return_value
        self.calls: list[dict] = []

    def get_subjects(self, **kwargs):
        self.calls.append(kwargs)
        return self.return_value


class TestGetSubjects:
    def test_returns_the_clients_dataframe(self):
        expected = pd.DataFrame({"name": ["Cherop"]})
        client = _FakeClient(expected)

        result = get_subjects(client=client)

        assert result is expected

    def test_ids_list_is_joined_into_a_comma_separated_id_kwarg(self):
        client = _FakeClient(pd.DataFrame({"name": ["a"]}))

        get_subjects(client=client, ids=["id-1", "id-2", "id-3"])

        assert client.calls[0]["id"] == "id-1,id-2,id-3"

    def test_no_ids_passes_id_as_none(self):
        client = _FakeClient(pd.DataFrame({"name": ["a"]}))

        get_subjects(client=client)

        assert client.calls[0]["id"] is None

    def test_filter_kwargs_are_forwarded_unchanged(self):
        client = _FakeClient(pd.DataFrame({"name": ["a"]}))

        get_subjects(
            client=client,
            include_inactive=True,
            bbox=(1.0, 2.0, 3.0, 4.0),
            subject_group_id="grp-1",
            subject_group_name="Elephants",
            name="Cherop",
            updated_since="2024-01-01",
            updated_until="2024-06-01",
            tracks=True,
            max_ids_per_request=25,
        )

        call = client.calls[0]
        assert call["include_inactive"] is True
        assert call["bbox"] == (1.0, 2.0, 3.0, 4.0)
        assert call["subject_group_id"] == "grp-1"
        assert call["subject_group_name"] == "Elephants"
        assert call["name"] == "Cherop"
        assert call["updated_since"] == "2024-01-01"
        assert call["updated_until"] == "2024-06-01"
        assert call["tracks"] is True
        assert call["max_ids_per_request"] == 25

    def test_raise_on_empty_true_and_empty_result_raises(self):
        client = _FakeClient(pd.DataFrame())

        with pytest.raises(ValueError, match="No data returned from EarthRanger"):
            get_subjects(client=client, raise_on_empty=True)

    def test_raise_on_empty_false_and_empty_result_returns_empty_df(self):
        client = _FakeClient(pd.DataFrame())

        result = get_subjects(client=client, raise_on_empty=False)

        assert result.empty

    def test_raise_on_empty_true_and_nonempty_result_does_not_raise(self):
        client = _FakeClient(pd.DataFrame({"name": ["Cherop"]}))

        result = get_subjects(client=client, raise_on_empty=True)

        assert not result.empty

"""Tests for ecoscope_workflows_ext_mep.tasks.results._persist.

`persist_subject_photo` is registered via `wt_registry.register()`, a
no-op decorator at call time, so it is called directly as plain Python.
`ecoscope.io.utils.download_file` is monkeypatched to write a small real
file to the target path (rather than hitting the network), so the rest of
the function -- filename derivation, deduplication, skip logic -- is
exercised against a real filesystem via `tmp_path`.

`safe_string` is not registered but is reused elsewhere in this package
(e.g. `reporting._subject_tracking`), so it is covered directly too.
"""

from __future__ import annotations

import pandas as pd
import pytest

import ecoscope_workflows_ext_mep.tasks.results._persist as persist_mod
from ecoscope_workflows_ext_mep.tasks.results._persist import (
    persist_subject_photo,
    safe_string,
)


@pytest.fixture(autouse=True)
def _fake_download_file(monkeypatch):
    """Replace `ecoscope.io.utils.download_file` with one that writes a
    trivial real file at `path`, recording every call made.
    """
    calls: list[tuple[str, str, bool]] = []

    def _fake(url: str, path: str, overwrite_existing: bool):
        calls.append((url, path, overwrite_existing))
        with open(path, "wb") as f:
            f.write(b"fake-image-bytes")

    monkeypatch.setattr(persist_mod.ecoscope.io.utils, "download_file", _fake)
    return calls


# --------------------------------------------------------------------------- #
# safe_string                                                                 #
# --------------------------------------------------------------------------- #


class TestSafeString:
    def test_spaces_become_underscores_and_result_is_lowercased(self):
        assert safe_string("Cherop The Elephant") == "cherop_the_elephant"

    def test_special_characters_are_stripped(self):
        assert safe_string("Chero!p@#$%") == "cherop"

    def test_leading_trailing_underscores_are_stripped(self):
        assert safe_string("  Cherop  ") == "cherop"

    def test_hyphens_are_preserved(self):
        assert safe_string("cherop-2") == "cherop-2"


# --------------------------------------------------------------------------- #
# persist_subject_photo                                                       #
# --------------------------------------------------------------------------- #


class TestPersistSubjectPhoto:
    def test_downloads_photo_named_after_filename_column(self, tmp_path, _fake_download_file):
        subject_df = pd.DataFrame({"photo": ["https://example.com/a.png"], "subject_name": ["Cherop"]})

        result = persist_subject_photo(
            subject_df=subject_df,
            column="photo",
            filename_column="subject_name",
            output_path=str(tmp_path),
        )

        assert result == [str(tmp_path / "cherop.png")]
        assert (tmp_path / "cherop.png").exists()

    def test_falls_back_to_url_hash_when_no_filename_column(self, tmp_path):
        subject_df = pd.DataFrame({"photo": ["https://example.com/a.png"]})

        result = persist_subject_photo(subject_df=subject_df, column="photo", output_path=str(tmp_path))

        assert len(result) == 1
        assert "profile_photo_" in result[0]

    def test_duplicate_filenames_are_disambiguated(self, tmp_path):
        subject_df = pd.DataFrame(
            {
                "photo": ["https://example.com/a.png", "https://example.com/b.png"],
                "subject_name": ["Cherop", "Cherop"],
            }
        )

        result = persist_subject_photo(
            subject_df=subject_df,
            column="photo",
            filename_column="subject_name",
            output_path=str(tmp_path),
        )

        assert sorted(result) == sorted([str(tmp_path / "cherop.png"), str(tmp_path / "cherop_1.png")])

    def test_missing_column_returns_empty_list(self, tmp_path):
        subject_df = pd.DataFrame({"other_col": ["x"]})

        result = persist_subject_photo(subject_df=subject_df, column="photo", output_path=str(tmp_path))

        assert result == []

    def test_invalid_url_row_is_skipped(self, tmp_path):
        subject_df = pd.DataFrame({"photo": ["not-a-url", "https://example.com/a.png"]})

        result = persist_subject_photo(subject_df=subject_df, column="photo", output_path=str(tmp_path))

        assert len(result) == 1

    def test_nan_value_row_is_skipped(self, tmp_path):
        subject_df = pd.DataFrame({"photo": [None, "https://example.com/a.png"]})

        result = persist_subject_photo(subject_df=subject_df, column="photo", output_path=str(tmp_path))

        assert len(result) == 1

    def test_dict_value_with_url_key_is_extracted(self, tmp_path):
        subject_df = pd.DataFrame({"photo": [{"url": "https://example.com/a.png"}]})

        result = persist_subject_photo(subject_df=subject_df, column="photo", output_path=str(tmp_path))

        assert len(result) == 1

    def test_dropbox_dl0_is_rewritten_to_dl1(self, tmp_path, _fake_download_file):
        subject_df = pd.DataFrame({"photo": ["https://www.dropbox.com/s/x/a.png?dl=0"]})

        persist_subject_photo(subject_df=subject_df, column="photo", output_path=str(tmp_path))

        assert _fake_download_file[0][0].endswith("dl=1")

    def test_image_type_without_leading_dot_is_normalized(self, tmp_path):
        subject_df = pd.DataFrame({"photo": ["https://example.com/a.jpg"], "subject_name": ["Cherop"]})

        result = persist_subject_photo(
            subject_df=subject_df,
            column="photo",
            filename_column="subject_name",
            output_path=str(tmp_path),
            image_type="jpg",
        )

        assert result[0].endswith(".jpg")

    def test_default_output_path_is_created_if_missing(self, tmp_path, monkeypatch):
        target = tmp_path / "nested" / "dir"
        monkeypatch.setattr(persist_mod.os, "getcwd", lambda: str(target))
        subject_df = pd.DataFrame({"photo": ["https://example.com/a.png"]})

        persist_subject_photo(subject_df=subject_df, column="photo", output_path=None)

        assert target.is_dir()

    def test_download_error_for_one_row_does_not_abort_others(self, tmp_path, monkeypatch):
        def _flaky(url, path, overwrite_existing):
            if "bad" in url:
                raise RuntimeError("network error")
            with open(path, "wb") as f:
                f.write(b"ok")

        monkeypatch.setattr(persist_mod.ecoscope.io.utils, "download_file", _flaky)
        subject_df = pd.DataFrame({"photo": ["https://example.com/bad.png", "https://example.com/good.png"]})

        result = persist_subject_photo(subject_df=subject_df, column="photo", output_path=str(tmp_path))

        # Only the "good" row's download succeeds; the "bad" row's exception
        # is caught and skipped rather than aborting the whole batch.
        assert len(result) == 1

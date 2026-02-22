import numpy as np
import pandas as pd
import pytest

from stochatreat.preparation import DataPreparator


@pytest.fixture
def simple_df():
    n = 10
    return pd.DataFrame({"id": np.arange(n), "stratum": np.arange(n) % 3})


class TestDataPreparator:
    def test_returns_stratum_id(self, simple_df):
        prep, idx_col = DataPreparator(
            simple_df, ["stratum"], "id", None, 42
        ).prepare()
        assert "stratum_id" in prep.columns
        assert idx_col == "id"

    def test_sorted_by_idx_col(self, simple_df):
        df = simple_df.sample(frac=1, random_state=0)
        prep, _ = DataPreparator(df, ["stratum"], "id", None, 42).prepare()
        assert list(prep["id"]) == sorted(prep["id"])

    def test_only_keeps_id_and_stratum_id(self, simple_df):
        prep, idx_col = DataPreparator(
            simple_df, ["stratum"], "id", None, 42
        ).prepare()
        assert list(prep.columns) == [idx_col, "stratum_id"]

    def test_empty_df(self):
        with pytest.raises(ValueError, match="not empty"):
            DataPreparator(
                pd.DataFrame(), ["stratum"], "id", None, 42
            ).prepare()

    def test_too_few_rows(self):
        df = pd.DataFrame({"id": [1], "stratum": ["a"]})
        with pytest.raises(ValueError, match="2 rows"):
            DataPreparator(df, ["stratum"], "id", None, 42).prepare()

    def test_idx_col_not_string(self, simple_df):
        with pytest.raises(TypeError, match="idx_col has to be a string"):
            DataPreparator(simple_df, ["stratum"], 0, None, 42).prepare()  # type: ignore

    def test_missing_stratum_col(self, simple_df):
        with pytest.raises(KeyError):
            DataPreparator(simple_df, ["not_a_col"], "id", None, 42).prepare()

    def test_missing_idx_col(self, simple_df):
        with pytest.raises(KeyError):
            DataPreparator(
                simple_df, ["stratum"], "not_a_col", None, 42
            ).prepare()

    def test_duplicate_idx_col(self):
        df = pd.DataFrame({"id": [1, 1, 2], "stratum": ["a", "a", "b"]})
        with pytest.raises(ValueError, match="not unique"):
            DataPreparator(df, ["stratum"], "id", None, 42).prepare()

    def test_size_too_large(self, simple_df):
        with pytest.raises(
            ValueError, match="larger than the sample universe"
        ):
            DataPreparator(simple_df, ["stratum"], "id", 999, 42).prepare()

    def test_size_samples_correctly(self):
        df = pd.DataFrame(
            {"id": np.arange(100), "stratum": np.arange(100) % 3}
        )
        size = 30
        prep, _ = DataPreparator(df, ["stratum"], "id", size, 42).prepare()
        assert len(prep) == size

    def test_none_idx_col_synthesised(self):
        df = pd.DataFrame({"stratum": ["a", "b", "c"]})
        prep, idx_col = DataPreparator(
            df, ["stratum"], None, None, 42
        ).prepare()
        assert idx_col == "index"
        assert "index" in prep.columns

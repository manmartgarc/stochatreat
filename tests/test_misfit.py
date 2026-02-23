import numpy as np
import pandas as pd
import pytest

from stochatreat.misfit import (
    GlobalMisfitHandler,
    NoneMisfitHandler,
    StratumMisfitHandler,
    _extract_misfits,
    make_misfit_handler,
)


@pytest.fixture
def odd_stratum_df():
    return pd.DataFrame(
        {"id": list(range(9)), "stratum_id": [0] * 5 + [1] * 4}
    )


class TestStratumMisfitHandler:
    def test_is_noop(self):
        df = pd.DataFrame({"id": [1, 2, 3], "stratum_id": [0, 0, 1]})
        result = StratumMisfitHandler().handle(df, lcm=2, random_state=0)
        pd.testing.assert_frame_equal(result, df)


class TestGlobalMisfitHandler:
    def test_moves_misfits_to_minus_one(self, odd_stratum_df):
        # lcm=2 -> stratum 0 has 1 misfit (5 % 2), stratum 1 has 0 misfits (4 % 2)
        result = GlobalMisfitHandler().handle(
            odd_stratum_df, lcm=2, random_state=0
        )
        assert -1 in result["stratum_id"].values
        assert (result["stratum_id"] == -1).sum() == 1

    def test_good_form_rows_unchanged(self):
        df = pd.DataFrame(
            {"id": list(range(7)), "stratum_id": [0] * 4 + [1] * 3}
        )
        # lcm=4 -> stratum 0: 0 misfits, stratum 1: 3 misfits
        result = GlobalMisfitHandler().handle(df, lcm=4, random_state=0)
        good = result[result["stratum_id"] != -1]
        expected_good = 4  # only stratum 0's 4 rows are misfit-free
        assert len(good) == expected_good
        assert set(good["stratum_id"]) == {0}


class TestNoneMisfitHandler:
    def test_marks_misfits_with_na(self, odd_stratum_df):
        result = NoneMisfitHandler().handle(
            odd_stratum_df, lcm=2, random_state=0
        )
        assert result["stratum_id"].isna().sum() == 1

    def test_converts_stratum_id_to_nullable_int(self):
        df = pd.DataFrame({"id": [1, 2, 3], "stratum_id": [0, 0, 1]})
        result = NoneMisfitHandler().handle(df, lcm=2, random_state=0)
        assert result["stratum_id"].dtype == "Int64"


class TestMakeMisfitHandler:
    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            ("stratum", StratumMisfitHandler),
            ("global", GlobalMisfitHandler),
            ("none", NoneMisfitHandler),
        ],
    )
    def test_returns_correct_type(self, strategy, expected):
        assert isinstance(make_misfit_handler(strategy), expected)


def test_extract_misfits_independent_randomness():
    n_strata = 100
    df = pd.DataFrame(
        {
            "id": range(n_strata * 10),
            "stratum_id": np.repeat(range(n_strata), 10),
        }
    )

    _, misfits = _extract_misfits(df, lcm=3, random_state=42)

    relative_positions = misfits["id"] % 10

    assert relative_positions.nunique() > 1, (
        "Misfits are sampled at the same relative position in every stratum!"
    )

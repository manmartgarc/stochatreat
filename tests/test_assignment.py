import uuid
import warnings

import numpy as np
import pandas as pd
import pytest

from stochatreat.stochatreat import stochatreat
from stochatreat.utils import get_lcm_prob_denominators

###############################################################################
# fixtures
###############################################################################


@pytest.fixture(params=[10_000, 100_000])
def df(request):
    n = request.param
    return pd.DataFrame(
        data={
            "id": np.arange(n),
            "dummy": [1] * n,
            "stratum1": np.random.randint(1, 100, size=n),
            "stratum2": np.random.randint(0, 2, size=n),
        }
    )


# a set of treatment assignment probabilities to throw at many tests
standard_probs = [
    [0.1, 0.9],
    [1 / 3, 2 / 3],
    [0.5, 0.5],
    [2 / 3, 1 / 3],
    [0.9, 0.1],
    [1 / 2, 1 / 3, 1 / 6],
    [
        14.28 / 100.0,
        42.86 / 100.0,
        42.86 / 100.0,
    ],  # approximates 1/7, 3/7, 3/7
]

# a set of stratum column combinations from the above df fixture to throw at
# many tests
standard_stratum_cols = [
    ["dummy"],
    ["stratum1"],
    ["stratum1", "stratum2"],
]


# a DataFrame and treatment assignment probabilities under which there will be
# no misfits
@pytest.fixture
def df_no_misfits():
    n = 1_000
    stratum_size = 10
    return pd.DataFrame(
        data={
            "id": np.arange(n),
            "stratum": np.repeat(
                np.arange(n / stratum_size), repeats=stratum_size
            ),
        }
    )


probs_no_misfits = [
    [0.1, 0.9],
    [0.5, 0.5],
    [0.9, 0.1],
]


###############################################################################
# overall treatment assignment proportions
###############################################################################


@pytest.mark.parametrize("n_treats", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_no_probs(n_treats, stratum_cols, df):
    treats = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=n_treats,
        idx_col="id",
        random_state=42,
    )

    treatment_shares = treats.groupby("treat")["id"].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array([1 / n_treats] * n_treats), decimal=2
    )


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_probs(probs, stratum_cols, df):
    treats = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby("treat")["id"].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array(probs), decimal=2
    )


@pytest.mark.parametrize("probs", probs_no_misfits)
def test_stochatreat_no_misfits(probs, df_no_misfits):
    treats = stochatreat(
        data=df_no_misfits,
        stratum_cols=["stratum"],
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby("treat")["id"].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array(probs), decimal=2
    )


@pytest.mark.parametrize("probs", standard_probs)
def test_stochatreat_only_misfits(probs):
    n = 10_000
    df = pd.DataFrame(
        data={
            "id": np.arange(n),
            "stratum": np.arange(n),
        }
    )
    treats = stochatreat(
        data=df,
        stratum_cols=["stratum"],
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby("treat")["id"].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array(probs), decimal=2
    )


###############################################################################
# within-stratum treatment assignments
###############################################################################


def get_within_strata_counts(treats):
    treatment_counts = (
        treats.groupby(["stratum_id", "treat"])[["id"]]
        .count()
        .rename(columns={"id": "treat_count"})
        .reset_index()
    )

    stratum_counts = (
        treats.groupby(["stratum_id"])[["id"]]
        .count()
        .rename(columns={"id": "stratum_count"})
        .reset_index()
    )

    return pd.merge(
        treatment_counts, stratum_counts, on="stratum_id", how="left"
    )


def compute_count_diff(treats, probs):
    counts = get_within_strata_counts(treats)

    required_props = pd.DataFrame(
        {"required_prop": probs, "treat": range(len(probs))}
    )
    comp = pd.merge(counts, required_props, on="treat", how="left")
    comp["desired_counts"] = comp["stratum_count"] * comp["required_prop"]

    comp["count_diff"] = (comp["treat_count"] - comp["desired_counts"]).abs()

    return comp


@pytest.mark.parametrize("n_treats", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_within_strata_no_probs(n_treats, stratum_cols, df):
    probs = n_treats * [1 / n_treats]
    lcm_prob_denominators = n_treats
    treats = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=n_treats,
        idx_col="id",
        random_state=42,
    )
    comp = compute_count_diff(treats, probs)

    assert_msg = """The counts differences exceed the bound that misfit
    allocation should not exceed"""
    assert (comp["count_diff"] < lcm_prob_denominators).all(), assert_msg


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_within_strata_probs(probs, stratum_cols, df):
    lcm_prob_denominators = get_lcm_prob_denominators(probs)
    treats = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    comp = compute_count_diff(treats, probs)

    assert_msg = """The counts differences exceed the bound that misfit
    allocation should not exceed"""
    assert (comp["count_diff"] < lcm_prob_denominators).all(), assert_msg


@pytest.mark.parametrize("probs", probs_no_misfits)
def test_stochatreat_within_strata_no_misfits(probs, df_no_misfits):
    treats = stochatreat(
        data=df_no_misfits,
        stratum_cols=["stratum"],
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    comp = compute_count_diff(treats, probs)

    assert_msg = "The required proportions are not reached without misfits"
    assert (comp["count_diff"] == 0).all(), assert_msg


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_global_strategy(probs, stratum_cols, df):
    treats = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
        misfit_strategy="global",
    )
    comp = compute_count_diff(treats, probs)

    stratum_count_diff = comp.groupby(["stratum_id"])["count_diff"].sum()

    assert_msg = "There is more than one stratum with misfits"
    assert (stratum_count_diff != 0).sum() <= 1, assert_msg


@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_stratum_ids(df, misfit_strategy, stratum_cols):
    treats = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=2,
        idx_col="id",
        random_state=42,
        misfit_strategy=misfit_strategy,
    )

    n_unique_strata = len(df[stratum_cols].drop_duplicates())

    n_unique_stratum_ids = len(treats["stratum_id"].drop_duplicates())

    if misfit_strategy == "global":
        # depending on whether there are misfits
        assert n_unique_strata in {
            n_unique_stratum_ids,
            n_unique_stratum_ids - 1,
        }
    else:
        assert n_unique_stratum_ids == n_unique_strata


@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
def test_stochatreat_random_state(df, stratum_cols, misfit_strategy):
    random_state = 42
    treats = []
    for _ in range(2):
        treatments_i = stochatreat(
            data=df,
            stratum_cols=stratum_cols,
            treats=2,
            idx_col="id",
            random_state=random_state,
            misfit_strategy=misfit_strategy,
        )
        treats.append(treatments_i)

    pd.testing.assert_series_equal(treats[0]["treat"], treats[1]["treat"])


@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
def test_stochatreat_shuffle_data(df, stratum_cols, misfit_strategy):
    random_state = 42
    treats = []
    for _ in range(2):
        treatments_i = stochatreat(
            data=df,
            stratum_cols=stratum_cols,
            treats=2,
            idx_col="id",
            random_state=random_state,
            misfit_strategy=misfit_strategy,
        )
        treatments_i = treatments_i.sort_values("id")
        treats.append(treatments_i)

        df = df.sample(len(df), random_state=random_state)

    pd.testing.assert_series_equal(treats[0]["treat"], treats[1]["treat"])


###############################################################################
# Miscellaneous tests
###############################################################################


def test_stochatreat_categorical_strata_warning():
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "stratum": pd.Categorical(["a", "b"], categories=["a", "b"]),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stochatreat(
            data=data,
            idx_col="id",
            stratum_cols=["stratum"],
            treats=2,
            probs=[1 / 2] * 2,
            random_state=42,
        )


def test_stochatreat_supports_big_integer_ids():
    # If we allow pandas to silently upcast, certain dtypes that convert to
    # float64 may lose precision, resulting in wrong or missing assignments:
    #   103241243500726324 is rounded to 103241243500726320
    #   2**53+1 is rounded to 2**53
    data = pd.DataFrame(
        {
            "id": [103241243500726324, 103241243500726320, 2**53, 2**53 + 1],
            "stratum": ["a", "a", "b", "b"],
        }
    )

    treatment_status = stochatreat(
        data=data,
        idx_col="id",
        stratum_cols=["stratum"],
        treats=2,
        probs=[1 / 2] * 2,
        random_state=42,
    )

    assert set(treatment_status["id"]) == set(data["id"])
    assert set(treatment_status["treat"]) == {0, 1}


def test_invalid_stratum_cols():
    df = pd.DataFrame({"id": [1, 2], "stratum": ["a", "b"]})
    with pytest.raises(KeyError):
        stochatreat(
            data=df,
            stratum_cols=["not_a_column"],
            treats=2,
            idx_col="id",
            probs=[0.5, 0.5],
        )


def test_output_column_order():
    df = pd.DataFrame({"id": [1, 2], "stratum": ["a", "b"]})
    result = stochatreat(
        data=df,
        stratum_cols=["stratum"],
        treats=2,
        idx_col="id",
        probs=[0.5, 0.5],
    )
    expected_order = ["id", "stratum_id", "treat"]
    assert list(result.columns) == expected_order


def test_duplicate_idx_col():
    df = pd.DataFrame({"id": [1, 1, 2], "stratum": ["a", "a", "b"]})
    with pytest.raises(
        ValueError, match="The values in idx_col are not unique"
    ):
        stochatreat(
            data=df,
            stratum_cols=["stratum"],
            treats=2,
            idx_col="id",
            probs=[0.5, 0.5],
        )


def test_invalid_misfit_strategy():
    df = pd.DataFrame({"id": [1, 2], "stratum": ["a", "b"]})
    with pytest.raises(ValueError, match="misfit_strategy must be one of"):
        stochatreat(
            data=df,
            stratum_cols=["stratum"],
            treats=2,
            idx_col="id",
            probs=[0.5, 0.5],
            misfit_strategy="not_a_strategy",  # type: ignore
        )


def test_idx_col_uuid_and_float():
    df = pd.DataFrame(
        {"id": [uuid.uuid4(), uuid.uuid4()], "stratum": ["a", "b"]}
    )
    result = stochatreat(
        data=df,
        stratum_cols=["stratum"],
        treats=2,
        idx_col="id",
        probs=[0.5, 0.5],
    )
    assert set(result["id"]) == set(df["id"])

    df_float = pd.DataFrame({"id": [1.1, 2.2], "stratum": ["a", "b"]})
    result_float = stochatreat(
        data=df_float,
        stratum_cols=["stratum"],
        treats=2,
        idx_col="id",
        probs=[0.5, 0.5],
    )
    assert set(result_float["id"]) == set(df_float["id"])


def test_stochatreat_crossplatform_flakiness():
    seed = 0
    rng = np.random.default_rng(seed)
    n = 100
    df = pd.DataFrame(
        {
            "id": range(n),
            "stratum": rng.choice(["a", "b"], n),
        }
    )
    assignments = stochatreat(
        data=df,
        stratum_cols=["stratum"],
        treats=2,
        idx_col="id",
        probs=[0.2, 0.8],
        random_state=42,
    )
    assert assignments["treat"].value_counts(ascending=True).tolist() == [
        21,
        79,
    ], f"assignments:\n{assignments.groupby(['stratum_id', 'treat']).count()}"

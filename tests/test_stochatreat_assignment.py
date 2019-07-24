import pytest

from math import gcd

import numpy as np
import pandas as pd

from stochatreat import stochatreat
from stochatreat import get_lcm_prob_denominators


################################################################################
# fixtures
################################################################################

@pytest.fixture(params=[10_000, 100_000])
def df(request):
    N = request.param
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "dummy": [1] * N,
            "stratum1": np.random.randint(1, 100, size=N),
            "stratum2": np.random.randint(0, 2, size=N),
        }
    )

    return df

# a set of treatment assignment probabilities to throw at many tests
standard_probs = [[0.1, 0.9],
                  [1/3, 2/3],
                  [0.5, 0.5],
                  [2/3, 1/3],
                  [0.9, 0.1]]

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
    N = 1_000
    stratum_size = 10
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "stratum": np.repeat(
                np.arange(N / stratum_size),
                repeats=stratum_size
            )
        }
    )

    return df

probs_no_misfits =[
    [0.1, 0.9],
    [0.5, 0.5],
    [0.9, 0.1],
]


################################################################################
# overall treatment assignment proportions
################################################################################

@pytest.mark.parametrize("n_treats", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_no_probs(n_treats, stratum_cols, df):
    """
    Tests that overall treatment assignment proportions across all strata are as
    intended with equal treatment assignment probabilities -- relies on the Law
    of Large Numbers, not deterministic
    """
    treats = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=n_treats,
        idx_col="id",
        random_state=42
    )

    treatment_shares = treats.groupby('treat')['id'].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array([1 / n_treats] * n_treats), decimal=2
    )


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_probs(probs, stratum_cols, df):
    """
    Tests that overall treatment assignment proportions across all strata are as
    intended with unequal treatment assignment probabilities -- relies on the
    Law of Large Numbers, not deterministic
    """
    treats = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby('treat')['id'].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array(probs), decimal=2
    )


@pytest.mark.parametrize("probs", probs_no_misfits)
def test_stochatreat_no_misfits(probs, df_no_misfits):
    """
    Tests that overall treatment assignment proportions across all strata are as
    intended when strata are such that there are no misfits
    """
    treats = stochatreat(
        data=df_no_misfits,
        stratum_cols=["stratum"],
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby('treat')['id'].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array(probs), decimal=2
    )


@pytest.mark.parametrize("probs", standard_probs)
def test_stochatreat_only_misfits(probs):
    """
    Tests that overall treatment assignment proportions across all strata are as
    intended when strata are such that there are only misfits and the number of
    units is sufficiently large -- relies on the Law of Large Numbers, not
    deterministic
    """
    N = 10_000
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "stratum": np.arange(N),
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
    treatment_shares = treats.groupby('treat')['id'].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array(probs), decimal=2
    )


################################################################################
# within-stratum treatment assignments
################################################################################

def get_within_strata_counts(treats):
    """Helper function to compute the treatment shares within strata"""
    treatment_counts = (treats
        .groupby(["stratum_id", "treat"])[["id"]]
        .count()
        .rename(columns={"id": "treat_count"})
        .reset_index()
    )

    stratum_counts = (treats
        .groupby(["stratum_id"])[["id"]]
        .count()
        .rename(columns={"id": "stratum_count"})
        .reset_index()
    )

    counts = pd.merge(
        treatment_counts, stratum_counts, on="stratum_id", how="left"
    )

    return counts


def compute_count_diff(treats, probs):
    """
    Helper function to compute the treatment counts within strata and line them
    up with required counts, and returns the different treatment counts
    aggregated at the stratum level as well as the dataframe with the different
    counts used in some tests
    """
    counts = get_within_strata_counts(treats)

    required_props = pd.DataFrame(
        {"required_prop": probs, "treat": range(len(probs))}
    )
    comp = pd.merge(
        counts, required_props, on="treat", how="left"
    )
    comp["desired_counts"] = comp["stratum_count"] * comp["required_prop"]

    comp["count_diff"] = (comp["treat_count"] - comp["desired_counts"]).abs()

    return comp


@pytest.mark.parametrize("n_treats", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_within_strata_no_probs(n_treats, stratum_cols, df):
    """
    Tests that within strata treatment assignment counts are only as far from
    the required counts as misfit assignment randomization allows with equal
    treatment assignment probabilities but a differing number of treatments
    """
    probs = n_treats * [1 / n_treats]
    lcm_prob_denominators = n_treats
    treats = stochatreat(
        data=df, 
        stratum_cols=stratum_cols, 
        treats=n_treats, 
        idx_col="id", 
        random_state=42
    )
    comp = compute_count_diff(treats, probs)

    assert_msg = """The counts differences exceed the bound that misfit 
    allocation should not exceed"""
    assert (comp["count_diff"] < lcm_prob_denominators).all(), assert_msg


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_within_strata_probs(probs, stratum_cols, df):
    """
    Tests that within strata treatment assignment counts are only as far from
    the required counts as misfit assignment randomization allows with two
    treatments but unequal treatment assignment probabilities
    """
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
    """
    Tests that within strata treatment assignment counts are exactly equal to
    the required counts when strata are such that there are no misfits
    """
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
        misfit_strategy="global"
    )
    comp = compute_count_diff(treats, probs)

    stratum_count_diff = comp.groupby(["stratum_id"])["count_diff"].sum()

    assert_msg = "There is more than one stratum with misfits"
    assert (stratum_count_diff != 0).sum() <= 1, assert_msg


@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
def test_stochatreat_stratum_ids(df, misfit_strategy, stratum_cols):
    """Tests that the function returns the right number of stratum ids"""
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
        assert (
            (n_unique_stratum_ids == n_unique_strata) or
            (n_unique_stratum_ids - 1 == n_unique_strata)
        )
    else:
        assert n_unique_stratum_ids == n_unique_strata


@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
def test_stochatreat_random_state(df, stratum_cols, misfit_strategy):
    """
    Tests that the results are the same on two consecutive calls with the same
    random state
    """
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
    
    pd.testing.assert_series_equal(
        treats[0]["treat"], treats[1]["treat"]
    )

    
@pytest.mark.parametrize("stratum_cols", standard_stratum_cols)
@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
def test_stochatreat_shuffle_data(df, stratum_cols, misfit_strategy):
    """
    Tests that the mapping between idx_col and the assignments is the same on
    two consecutive calls with the same random state and shuffled data points
    """
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
    
    pd.testing.assert_series_equal(
        treats[0]["treat"], treats[1]["treat"]
    )




    




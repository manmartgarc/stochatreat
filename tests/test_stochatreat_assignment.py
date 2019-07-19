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
            "block1": np.random.randint(1, 100, size=N),
            "block2": np.random.randint(0, 2, size=N),
        }
    )

    return df

# a set of treatment assignment probabilities to throw at many tests
standard_probs = [[0.1, 0.9],
                  [1/3, 2/3],
                  [0.5, 0.5],
                  [2/3, 1/3],
                  [0.9, 0.1]]

# a set of block column combinations from the above df fixture to throw at many tests
standard_block_cols = [
    ["dummy"],
    ["block1"],
    ["block1", "block2"],
]


# a DataFrame and treatment assignment probabilities under which there will be no misfits
@pytest.fixture
def df_no_misfits():
    N = 1_000
    blocksize = 10
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "block": np.repeat(
                np.arange(N / blocksize),
                repeats=blocksize
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
@pytest.mark.parametrize("block_cols", standard_block_cols)
def test_stochatreat_no_probs(n_treats, block_cols, df):
    """
    Tests that overall treatment assignment proportions across all strata are as intended
    with equal treatment assignment probabilities
    -- relies on the Law of Large Numbers, not deterministic
    """
    treats = stochatreat(
        data=df,
        block_cols=block_cols,
        treats=n_treats,
        idx_col="id",
        random_state=42
    )

    treatment_shares = treats.groupby('treat')['id'].size() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array([1 / n_treats] * n_treats), decimal=2
    )


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize("block_cols", standard_block_cols)
def test_stochatreat_probs(probs, block_cols, df):
    """
    Tests that overall treatment assignment proportions across all strata are as intended
    with unequal treatment assignment probabilities
    -- relies on the Law of Large Numbers, not deterministic
    """
    treats = stochatreat(
        data=df,
        block_cols=block_cols,
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
    Tests that overall treatment assignment proportions across all strata are as intended
    when strata are such that there are no misfits
    """
    treats = stochatreat(
        data=df_no_misfits,
        block_cols=["block"],
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
    Tests that overall treatment assignment proportions across all strata are as intended
    when strata are such that there are only misfits and the number of units is
    sufficiently large
    -- relies on the Law of Large Numbers, not deterministic
    """
    N = 10_000
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "block": np.arange(N),
        }
    )
    treats = stochatreat(
        data=df,
        block_cols=["block"],
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
        .groupby(["block_id", "treat"])[["id"]]
        .count()
        .rename(columns={"id": "treat_count"})
        .reset_index()
    )

    block_counts = (treats
        .groupby(["block_id"])[["id"]]
        .count()
        .rename(columns={"id": "block_count"})
        .reset_index()
    )

    counts = pd.merge(
        treatment_counts, block_counts, on="block_id", how="left"
    )

    return counts


def compute_count_diff(treats, probs):
    """
    Helper function to compute the treatment counts within strata and line them up
    with required counts, and returns the different treatment counts aggregated at the block level
    as well as the dataframe with the different counts used in some tests
    """
    counts = get_within_strata_counts(treats)

    required_props = pd.DataFrame({"required_prop": probs, "treat": range(len(probs))})
    comparison_df = pd.merge(
        counts, required_props, on="treat", how="left"
    )
    comparison_df["desired_counts"] = comparison_df["block_count"] * comparison_df["required_prop"]

    comparison_df["count_diff"] = (comparison_df["treat_count"] - comparison_df["desired_counts"]).abs()

    return comparison_df


@pytest.mark.parametrize("n_treats", [2, 3, 4, 5, 10])
@pytest.mark.parametrize(
    "block_cols", standard_block_cols
)
def test_stochatreat_within_strata_no_probs(n_treats, block_cols, df):
    """
    Tests that within strata treatment assignment counts are only as far from the required
    counts as misfit assignment randomization allows with equal treatment assignment
    probabilities but a differing number of treatments
    """
    probs = n_treats * [1 / n_treats]
    lcm_prob_denominators = n_treats
    treats = stochatreat(
        data=df, block_cols=block_cols, treats=n_treats, idx_col="id", random_state=42
    )
    comparison_df = compute_count_diff(treats, probs)

    assert_msg = "The counts differences exceed the bound that misfit allocation should not exceed"
    assert (comparison_df["count_diff"] < lcm_prob_denominators).all(), assert_msg


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize(
    "block_cols", standard_block_cols
)
def test_stochatreat_within_strata_probs(probs, block_cols, df):
    """
    Tests that within strata treatment assignment counts are only as far from the required
    counts as misfit assignment randomization allows with two treatments but unequal
    treatment assignment probabilities
    """
    lcm_prob_denominators = get_lcm_prob_denominators(probs)
    treats = stochatreat(
        data=df,
        block_cols=block_cols,
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    comparison_df = compute_count_diff(treats, probs)

    assert_msg = "The counts differences exceed the bound that misfit allocation should not exceed"
    assert (comparison_df["count_diff"] < lcm_prob_denominators).all(), assert_msg


@pytest.mark.parametrize("probs", probs_no_misfits)
def test_stochatreat_within_strata_no_misfits(probs, df_no_misfits):
    """
    Tests that within strata treatment assignment counts are exactly equal to the required
    counts when strata are such that there are no misfits
    """
    treats = stochatreat(
        data=df_no_misfits,
        block_cols=['block'],
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    comparison_df = compute_count_diff(treats, probs)

    assert_msg = "The required proportions are not reached without misfits"
    assert (comparison_df["count_diff"] == 0).all(), assert_msg


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize(
    "block_cols", standard_block_cols
)
def test_stochatreat_global_strategy(probs, block_cols, df):
    treats = stochatreat(
        data=df,
        block_cols=block_cols,
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
        misfit_strategy="global"
    )
    comparison_df = compute_count_diff(treats, probs)

    block_count_diff = comparison_df.groupby(["block_id"])["count_diff"].sum()

    assert_msg = "There is more than one block with misfits"
    assert (block_count_diff != 0).sum() <= 1, assert_msg


@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
@pytest.mark.parametrize(
    "block_cols", standard_block_cols
    )
def test_stochatreat_block_ids(df, misfit_strategy, block_cols):
    """Tests that the function returns the right number of block ids"""
    treats = stochatreat(
        data=df,
        block_cols=block_cols,
        treats=2,
        idx_col="id",
        random_state=42,
        misfit_strategy=misfit_strategy,
    )

    n_unique_blocks = len(df[block_cols].drop_duplicates())

    n_unique_block_ids = len(treats["block_id"].drop_duplicates())

    if misfit_strategy == "global":
        # depending on whether there are misfits
        assert (n_unique_block_ids == n_unique_blocks) | (n_unique_block_ids - 1 == n_unique_blocks)
    else:
        assert n_unique_block_ids == n_unique_blocks

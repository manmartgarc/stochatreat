import pytest

import numpy as np
import pandas as pd

from stochatreat import stochatreat

np.random.seed(42)

standard_probs = [[0.1, 0.9], [1/3, 2/3], [0.5, 0.5], [2/3, 1/3], [0.9, 0.1]]


@pytest.fixture(params=[10000, 100000])
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


@pytest.mark.parametrize("n_treats", [2, 3, 4, 5, 10])
@pytest.mark.parametrize(
    "block_cols", [["dummy"], ["block1"], ["block1", "block2"]]
)
def test_stochatreat_no_probs(n_treats, block_cols, df):
    """Test that overall treatment assignment proportions across all strata are as intended with equal treatment assignment probabilities"""
    treats = stochatreat(
        data=df, block_cols=block_cols, treats=n_treats, idx_col="id", random_state=42
    )

    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(
        treatment_shares, np.array([1 / n_treats] * n_treats), decimal=3
    )


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize(
    "block_cols", [["dummy"], ["block1"], ["block1", "block2"]]
)
def test_stochatreat_probs(probs, block_cols, df):
    """Test that overall treatment assignment proportions across all strata are as intended with unequal treatment assignment probabilities"""
    treats = stochatreat(
        data=df,
        block_cols=block_cols,
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(treatment_shares, np.array(probs), decimal=3)


@pytest.mark.parametrize("probs", [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
def test_stochatreat_no_misfits(probs):
    """Test that overall treatment assignment proportions across all strata are as intended when strata are such that there are no misfits"""
    N = 1_000_000
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

    treats = stochatreat(
        data=df,
        block_cols=['block'],
        treats=len(probs),
        idx_col="id",
        probs=probs,
        random_state=42,
    )
    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(treatment_shares, np.array(probs), decimal=3)


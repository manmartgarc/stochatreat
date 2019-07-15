import pytest

import numpy as np
import pandas as pd

from stochatreat import stochatreat


@pytest.fixture
def correct_params():
    """A set of valid parameters that can be passed to stochatreat()"""
    params = {
        "probs": [0.1, 0.9],
        "treat": 2,
        "data": pd.DataFrame(data={"id": np.arange(100), "block": np.arange(100)}),
        "idx_col": "id",
    }
    return params


def test_stochatreat_input_invalid_probs(correct_params):
    """Tests that the function rejects probabilities that don't add up to one"""
    probs_not_sum_to_one = [0.1, 0.2]
    with pytest.raises(Exception):
        stochatreat(
            data=correct_params["data"],
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=probs_not_sum_to_one,
        )


def test_stochatreat_input_more_treats_than_probs(correct_params):
    """Tests that the function raises an error for treatments and probs of different sizes"""
    treat_too_large = 3
    with pytest.raises(Exception):
        stochatreat(
            data=correct_params["data"],
            block_cols=["block"],
            treats=treat_too_large,
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


def test_stochatreat_input_empty_data(correct_params):
    """Tests that the function raises an error when an empty dataframe is passed"""
    empty_data = pd.DataFrame({})
    with pytest.raises(ValueError):
        stochatreat(
            data=empty_data,
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


def test_stochatreat_input_idx_col_str(correct_params):
    """Tests that the function rejects an idx_col parameter that is not a string or None"""
    idx_col_not_str = 0
    with pytest.raises(TypeError):
        stochatreat(
            data=correct_params["data"],
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=idx_col_not_str,
            probs=correct_params["probs"],
        )


def test_stochatreat_input_invalid_size(correct_params):
    """Tests that the function rejects a sampling size larger than the data count"""
    size_bigger_than_sampling_universe_size = 101
    with pytest.raises(ValueError):
        stochatreat(
            data=correct_params["data"],
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
            size=size_bigger_than_sampling_universe_size,
        )


def test_stochatreat_input_idx_col_unique(correct_params):
    """Tests that the function raises an error if the idx_col is not a primary key of the data"""
    data_with_idx_col_with_duplicates = pd.DataFrame(
        data={"id": 1, "block": np.arange(100)}
    )
    with pytest.raises(ValueError):
        stochatreat(
            data=data_with_idx_col_with_duplicates,
            block_cols=["block"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


@pytest.fixture
def treatments_dict():
    """fixture of stochatreat() output to test output format"""
    treats = 2
    data = pd.DataFrame(
        data={"id": np.arange(100), "block": [0] * 40 + [1] * 30 + [2] * 30}
    )
    idx_col = "id"
    size = 90

    treatments = stochatreat(
        data=data,
        block_cols=["block"],
        treats=treats,
        idx_col=idx_col,
        size=size,
        random_state=42,
    )

    treatments_dict = {
        "data": data,
        "idx_col": idx_col,
        "size": size,
        "treatments": treatments,
    }

    return treatments_dict


def test_stochatreat_output_type(treatments_dict):
    """Tests that the function's output is a pd DataFrame"""
    treatments_df = treatments_dict["treatments"]
    assert isinstance(treatments_df, pd.DataFrame), "The output is not a DataFrame"


def test_stochatreat_output_treat_col(treatments_dict):
    """Tests that the function's output contains the `treat` column"""
    treatments_df = treatments_dict["treatments"]
    assert "treat" in treatments_df.columns, "Treatment column is missing"


def test_stochatreat_output_treat_col_dtype(treatments_dict):
    """Tests that the function's output's 'treat` column is an int column"""
    treatments_df = treatments_dict["treatments"]
    assert treatments_df["treat"].dtype == np.int64, "Treatment column is missing"


def test_stochatreat_output_block_id_col(treatments_dict):
    """Tests that the function's output contains the `block_id`'"""
    treatments_df = treatments_dict["treatments"]
    assert "block_id" in treatments_df.columns, "Block_id column is missing"


def test_stochatreat_output_block_id_col_dtype(treatments_dict):
    """Tests that the function's output's 'block_id` column is an int column'"""
    treatments_df = treatments_dict["treatments"]
    assert treatments_df["block_id"].dtype == np.int64, "Block_id column is missing"


def test_stochatreat_output_idx_col(treatments_dict):
    """Tests that the function's output's 'idx_col` column is the same type as the input'"""
    treatments_df = treatments_dict["treatments"]
    data = treatments_dict["data"]
    idx_col = treatments_dict["idx_col"]
    assert treatments_df[idx_col].dtype == data[idx_col].dtype, "Index column is missing"


def test_stochatreat_output_size(treatments_dict):
    """Tests that the function's output is of the right length'"""
    treatments_df = treatments_dict["treatments"]
    size = treatments_dict["size"]
    assert len(treatments_df) == size, "The size of the output does not match the sampled size"


def test_stochatreat_output_no_null_treats(treatments_dict):
    """Tests that the function's output treatments are all non null'"""
    treatments_df = treatments_dict["treatments"]
    assert treatments_df["treat"].isnull().sum() == 0, "There are null assignments"


standard_probs = [[0.1, 0.9], [1 / 3, 2 / 3], [0.5, 0.5], [2 / 3, 1 / 3], [0.9, 0.1]]


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
@pytest.mark.parametrize("block_cols", [["dummy"], ["block1"], ["block1", "block2"]])
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
@pytest.mark.parametrize("block_cols", [["dummy"], ["block1"], ["block1", "block2"]])
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
    N = 10_000
    blocksize = 10
    df = pd.DataFrame(
        data={
            "id": np.arange(N),
            "block": np.repeat(np.arange(N / blocksize), repeats=blocksize),
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
    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(treatment_shares, np.array(probs), decimal=3)


@pytest.mark.parametrize("probs", standard_probs)
def test_stochatreat_only_misfits(probs):
    """Test that overall treatment assignment proportions across all strata are as intended when strata are such that there are only misfits"""
    N = 1_000
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
    treatment_shares = treats.groupby(["treat"])["id"].count() / treats.shape[0]

    np.testing.assert_almost_equal(treatment_shares, np.array(probs), decimal=3)


@pytest.mark.parametrize(
    "block_cols", [["dummy"], ["block1"], ["block1", "block2"]]
)
def test_stochatreat_block_ids(df, block_cols):
    """Tests that the function returns the right number of block ids"""
    treats = stochatreat(
        data=df,
        block_cols=block_cols,
        treats=2,
        idx_col="id",
        random_state=42,
    )

    n_unique_blocks = len(df[block_cols].drop_duplicates())

    n_unique_block_ids = len(treats["block_id"].drop_duplicates())

    np.testing.assert_equal(n_unique_block_ids, n_unique_blocks)

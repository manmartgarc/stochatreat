import numpy as np
import pandas as pd
import pytest
from stochatreat.stochatreat import stochatreat


@pytest.fixture
def correct_params():
    """
    A set of valid parameters that can be passed to stochatreat()
    """
    params = {
        "probs": [0.1, 0.9],
        "treat": 2,
        "data": pd.DataFrame(
            data={"id": np.arange(100), "stratum": np.arange(100)}
        ),
        "idx_col": "id",
    }
    return params


def test_stochatreat_input_invalid_probs(correct_params):
    """
    Tests that the function rejects probabilities that don't add up to one
    """
    probs_not_sum_to_one = [0.1, 0.2]
    with pytest.raises(Exception):
        stochatreat(
            data=correct_params["data"],
            stratum_cols=["stratum"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=probs_not_sum_to_one,
        )


def test_stochatreat_input_more_treats_than_probs(correct_params):
    """
    Tests that the function raises an error for treatments and probs of
    different sizes
    """
    treat_too_large = 3
    with pytest.raises(Exception):
        stochatreat(
            data=correct_params["data"],
            stratum_cols=["stratum"],
            treats=treat_too_large,
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


def test_stochatreat_input_empty_data(correct_params):
    """
    Tests that the function raises an error when an empty dataframe is passed
    """
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        stochatreat(
            data=empty_data,
            stratum_cols="stratum",
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


def test_stochatreat_input_idx_col_str(correct_params):
    """
    Tests that the function rejects an idx_col parameter that is not a
    string or None
    """
    idx_col_not_str = 0
    with pytest.raises(TypeError):
        stochatreat(
            data=correct_params["data"],
            stratum_cols=["stratum"],
            treats=correct_params["treat"],
            idx_col=idx_col_not_str,
            probs=correct_params["probs"],
        )


def test_stochatreat_input_invalid_size(correct_params):
    """
    Tests that the function rejects a sampling size larger than the data count
    """
    size_bigger_than_sampling_universe_size = 101
    with pytest.raises(ValueError):
        stochatreat(
            data=correct_params["data"],
            stratum_cols=["stratum"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
            size=size_bigger_than_sampling_universe_size,
        )


def test_stochatreat_input_idx_col_unique(correct_params):
    """
    Tests that the function raises an error if the idx_col is not a primary key
    of the data
    """
    data_with_idx_col_with_duplicates = pd.DataFrame(
        data={"id": 1, "stratum": np.arange(100)}
    )
    with pytest.raises(ValueError):
        stochatreat(
            data=data_with_idx_col_with_duplicates,
            stratum_cols=["stratum"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
        )


def test_stochatreat_input_invalid_strategy(correct_params):
    """
    Tests that the function raises an error if an invalid strategy string is
    passed
    """
    unknown_strategy = "unknown"
    with pytest.raises(ValueError):
        stochatreat(
            data=correct_params["data"],
            stratum_cols=["stratum"],
            treats=correct_params["treat"],
            idx_col=correct_params["idx_col"],
            probs=correct_params["probs"],
            misfit_strategy=unknown_strategy,
        )


@pytest.fixture
def treatments_dict():
    """fixture of stochatreat() output to test output format"""
    treats = 2
    data = pd.DataFrame(
        data={"id": np.arange(100), "stratum": [0] * 40 + [1] * 30 + [2] * 30}
    )
    idx_col = "id"
    size = 90

    treatments = stochatreat(
        data=data,
        stratum_cols=["stratum"],
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
    """
    Tests that the function's output is a pd DataFrame
    """
    treatments_df = treatments_dict["treatments"]
    assert_msg = "The output is not a DataFrame"
    assert isinstance(treatments_df, pd.DataFrame), assert_msg


def test_stochatreat_output_treat_col(treatments_dict):
    """
    Tests that the function's output contains the `treat` column
    """
    treatments_df = treatments_dict["treatments"]
    assert_msg = "Treatment column is missing"
    assert "treat" in treatments_df.columns, assert_msg


def test_stochatreat_output_treat_col_dtype(treatments_dict):
    """
    Tests that the function's output's 'treat` column is an int column
    """
    treatments_df = treatments_dict["treatments"]
    assert_msg = "Treatment column is missing"
    assert treatments_df["treat"].dtype == np.int64, assert_msg


def test_stochatreat_output_stratum_id_col(treatments_dict):
    """
    Tests that the function's output contains the `stratum_id`
    """
    treatments_df = treatments_dict["treatments"]
    assert_msg = "stratum_id column is missing"
    assert "stratum_id" in treatments_df.columns, assert_msg


def test_stochatreat_output_stratum_id_col_dtype(treatments_dict):
    """
    Tests that the function's output's 'stratum_id` column is an int column
    """
    treatments_df = treatments_dict["treatments"]
    assert_msg = "stratum_id column is missing"
    assert treatments_df["stratum_id"].dtype == np.int64, assert_msg


def test_stochatreat_output_idx_col(treatments_dict):
    """
    Tests that the function's output's 'idx_col` column is the same type as the
    input's
    """
    treatments_df = treatments_dict["treatments"]
    data = treatments_dict["data"]
    idx_col = treatments_dict["idx_col"]
    assert_msg = "Index column is missing"
    assert treatments_df[idx_col].dtype == data[idx_col].dtype, assert_msg


def test_stochatreat_output_size(treatments_dict):
    """
    Tests that the function's output is of the right length
    """
    treatments_df = treatments_dict["treatments"]
    size = treatments_dict["size"]
    assert_msg = "The size of the output does not match the sampled size"
    assert len(treatments_df) == size, assert_msg


def test_stochatreat_output_no_null_treats(treatments_dict):
    """
    Tests that the function's output treatments are all non null
    """
    treatments_df = treatments_dict["treatments"]
    assert_msg = "There are null assignments"
    assert treatments_df["treat"].isnull().sum() == 0, assert_msg


@pytest.fixture
def treatments_dict_rand_index():
    """fixture of stochatreat() output to test output format"""
    treats = 2
    data = pd.DataFrame(
        data={
            "id": np.random.permutation(100),
            "stratum": [0] * 40 + [1] * 30 + [2] * 30,
        }
    )
    data = data.set_index(pd.Index(np.random.choice(300, 100, replace=False)))
    idx_col = "id"

    treatments = stochatreat(
        data=data,
        stratum_cols=["stratum"],
        treats=treats,
        idx_col=idx_col,
        random_state=42,
    )

    treatments_dict = {
        "data": data,
        "stratum_cols": ["stratum"],
        "idx_col": idx_col,
        "treatments": treatments,
        "n_treatments": treats,
    }

    return treatments_dict


standard_probs = [
    [0.1, 0.9],
    [1 / 3, 2 / 3],
    [0.5, 0.5],
    [2 / 3, 1 / 3],
    [0.9, 0.1],
]


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
def test_stochatreat_output_index_content_unchanged(
    treatments_dict_rand_index, probs, misfit_strategy
):
    """
    Tests that the functions's output's index column matches the input index
    column
    """
    data_with_rand_index = treatments_dict_rand_index["data"]

    treatments = stochatreat(
        data=data_with_rand_index,
        stratum_cols=["stratum"],
        probs=probs,
        treats=2,
        idx_col=treatments_dict_rand_index["idx_col"],
        misfit_strategy=misfit_strategy,
    )

    assert_msg = "The output and input indices do not have the same content"
    assert set(treatments.index) == set(data_with_rand_index.index), assert_msg


@pytest.mark.parametrize("probs", standard_probs)
@pytest.mark.parametrize("misfit_strategy", ["global", "stratum"])
def test_stochatreat_output_index_and_idx_col_correspondence(
    treatments_dict_rand_index, probs, misfit_strategy
):
    """
    Tests that the functions's output's index column matches the input index
    column
    """
    data_with_rand_index = treatments_dict_rand_index["data"]
    idx_col = treatments_dict_rand_index["idx_col"]

    treatments = stochatreat(
        data=data_with_rand_index,
        stratum_cols="stratum",
        probs=probs,
        treats=2,
        idx_col=idx_col,
        misfit_strategy=misfit_strategy,
    )

    data_with_rand_index = data_with_rand_index.sort_index()
    treatments = treatments.sort_index()

    pd.testing.assert_series_equal(
        data_with_rand_index[idx_col], treatments[idx_col]
    )


def test_stochatreat_output_sample(correct_params):
    """
    Tests that the function samples to the correct size
    """
    size = 100
    assignments = stochatreat(
        data=correct_params["data"],
        stratum_cols=["stratum"],
        treats=correct_params["treat"],
        idx_col=correct_params["idx_col"],
        probs=correct_params["probs"],
        size=size,
    )

    assert len(assignments) == size

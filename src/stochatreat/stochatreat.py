"""Stratified random assignment of treatments to units.

This module provides a function to assign treatments to units in a
stratified manner. The function is designed to work with pandas
dataframes and is able to handle multiple strata. There are also different
strategies to deal with misfits (units that are left over after the
stratified assignment procedure).
"""

from __future__ import annotations

import math
from typing import Literal, get_args

import numpy as np
import pandas as pd

from stochatreat.utils import get_lcm_prob_denominators

MIN_ROW_N = 2
MisfitStrategy = Literal["stratum", "global"]


def stochatreat(
    data: pd.DataFrame,
    stratum_cols: list[str],
    treats: int,
    probs: list[float] | None = None,
    random_state: int = 42,
    idx_col: str | None = None,
    size: int | None = None,
    misfit_strategy: MisfitStrategy = "stratum",
) -> pd.DataFrame:
    """
    Takes a dataframe and an arbitrary number of treatments over an
    arbitrary number of strata.

    Attempts to return equally sized treatment groups, while randomly
    assigning misfits (left overs from strata not divisible by the number
    of treatments).

    Parameters
    ----------
    data            :   The data that contains unique ids and the
                        stratification columns.
    stratum_cols    :   The columns in 'data' that you want to stratify over.
    treats          :   The number of treatments you would like to
                        implement, including control.
    probs           :   The assignment probabilities for each of the
                        treatments.
    random_state    :   The seed for the rng instance.
    idx_col         :   The column name that indicates the ids for your data.
    size            :   The size of the sample if you would like to sample
                        from your data.
    misfit_strategy :   The strategy used to assign misfits. Can be one of
                        'stratum' or 'global'.
                        If 'stratum', will assign misfits randomly and
                        independently within each stratum using probs.
                        If 'global', will group all misfits into one stratum
                        and do a full assignment procedure in this new stratum
                        with local random assignments of the misfits in this
                        stratum

    Returns
    -------
    pandas.DataFrame with idx_col, treat (treatment assignments) and
    stratum_id (the id of the stratum within which the assignment procedure
    was carried out) columns

    Usage
    -----
    Single stratum:
        >>> treats = stochatreat(data=data,               # your dataframe
                                 stratum_cols='stratum1', # stratum variable
                                 treats=2,                # including control
                                 idx_col='myid',          # unique id column
                                 random_state=42)         # seed for rng
        >>> data = data.merge(treats, how="left", on="myid")

    Multiple strata:
        >>> treats = stochatreat(data=data,
                                 stratum_cols=['stratum1', 'stratum2'],
                                 treats=2,
                                 probs=[1/3, 2/3],
                                 idx_col='myid',
                                 random_state=42)
        >>> data = data.merge(treats, how="left", on="myid")
    """
    # we also do a runtime check for the naughty folks that don't use typing
    if misfit_strategy not in get_args(MisfitStrategy):
        msg = f"misfit_strategy must be one of {get_args(MisfitStrategy)}."
        raise ValueError(msg)
    rand = np.random.RandomState(random_state)

    # =========================================================================
    # do checks
    # =========================================================================
    # data = data.copy()

    # create treatment array and probability array
    treatment_ids = list(range(treats))
    # if no probabilities stated
    if probs is None:
        frac = 1 / len(treatment_ids)
        probs_np = np.array([frac] * len(treatment_ids))
    elif probs is not None:
        probs_np = np.array(probs)
        if not math.isclose(probs_np.sum(), 1, rel_tol=1e-9):
            error_msg = "The probabilities must add up to 1"
            raise ValueError(error_msg)
        if len(probs_np) != len(treatment_ids):
            error_msg = (
                "The number of probabilities must match the number of "
                "treatments"
            )
            raise ValueError(error_msg)

    # check if dataframe is empty
    if data.empty:
        error_msg = "Make sure that your dataframe is not empty."
        raise ValueError(error_msg)

    # check length of data
    if len(data) < MIN_ROW_N:
        error_msg = "Your dataframe at least needs to have 2 rows."
        raise ValueError(error_msg)

    # if idx_col parameter was not defined.
    if idx_col is None:
        data = data.rename_axis("index", axis="index").reset_index()
        idx_col = "index"
    elif not isinstance(idx_col, str):
        error_msg = "idx_col has to be a string."
        raise TypeError(error_msg)

    # retrieve type to check and re-assign in the end
    idx_col_type = data[idx_col].dtype

    # check for unique identifiers
    if data[idx_col].duplicated(keep=False).sum() > 0:
        error_msg = "The values in idx_col are not unique."
        raise ValueError(error_msg)

    # if size is larger than sample universe
    if size is not None and size > len(data):
        error_msg = "Size argument is larger than the sample universe."
        raise ValueError(error_msg)

    # deal with multiple strata
    if isinstance(stratum_cols, str):
        stratum_cols = [stratum_cols]

    # sort data - useful to preserve correspondence between `idx_col` and
    # assignments
    data = data.sort_values(by=idx_col)

    # combine strata cells - by assigning stratum ids
    data["stratum_id"] = data.groupby(stratum_cols, observed=False).ngroup()

    # keep only ids and concatenated strata
    data = data[[idx_col, "stratum_id"]].copy()

    # apply weights to each stratum if sampling is wanted
    if size is not None:
        size = int(size)
        # get sampling weights
        strata_fracs = (
            data["stratum_id"].value_counts(normalize=True).sort_index()
        )
        reduced_sizes = (strata_fracs * size).round().astype(int)
        # draw sample
        data = data.groupby("stratum_id").apply(
            lambda x: x.sample(
                n=reduced_sizes[x.name], random_state=random_state
            ),
            include_groups=False,
        )
        data["stratum_id"] = data.index.get_level_values(0)
        data = data.droplevel(level="stratum_id")

    # Treatment assignment proceeds in two stages within each stratum:
    # 1. In as far as units can be neatly divided in the proportions given by
    #    prob they are so divided.
    # 2. Any leftovers ("misfits") are dealt with using either of the methods
    #    described in the docstring

    # 1. determine how to divide cleanly as much as possible

    # convert all probs to fractions and get the lowest common multiple of
    # their denominators
    lcm_prob_denominators = get_lcm_prob_denominators(probs_np)

    # produce the assignment mask that we will use to achieve perfect
    # proportions
    treat_mask = np.repeat(
        treatment_ids, (lcm_prob_denominators * probs_np).astype(int)
    )

    # =========================================================================
    # re-arrange strata
    # =========================================================================

    if misfit_strategy == "global":
        # separate the global misfits
        misfit_data = data.groupby("stratum_id").apply(
            lambda x: x.sample(
                n=(x.shape[0] % lcm_prob_denominators),
                replace=False,
                random_state=random_state,
            ),
            include_groups=False,
        )
        misfit_data["stratum_id"] = misfit_data.index.get_level_values(0)
        misfit_data = misfit_data.droplevel(level="stratum_id")
        good_form_data = data.drop(index=misfit_data.index)

        # assign the misfits their own stratum and concatenate
        misfit_data.loc[:, "stratum_id"] = -1
        data = pd.concat([good_form_data, misfit_data])

    # =========================================================================
    # assign treatments
    # =========================================================================

    # sort by strata first, and assign a long list of permuted `treat_mask` to
    # deal with misfits, we add fake rows to each stratum so that its length is
    # divisible by `lcm_prob_denominators` and toss them later
    # -> no costly apply inside the strata

    # add fake rows for each stratum so the total number can be divided by
    # `lcm_prob_denominators`
    fake = pd.DataFrame({"fake": data.groupby("stratum_id").size()})
    fake = fake.reset_index()
    fake.loc[:, "fake"] = (
        lcm_prob_denominators - fake["fake"] % lcm_prob_denominators
    ) % lcm_prob_denominators
    fake_rep = pd.DataFrame(
        fake.values.repeat(fake["fake"], axis=0), columns=fake.columns
    )
    # Before we add fake data, protect the idx_col values from being upcasted
    # to a different type and mutating the original data due to the
    # introduction of nulls. We will restore the original type later.
    data[idx_col] = data[idx_col].astype(object)
    data.loc[:, "fake"] = 0
    fake_rep.loc[:, "fake"] = 1

    data = pd.concat([data, fake_rep], sort=False).sort_values(by="stratum_id")

    # generate random permutations without loop by generating large number of
    # random values and sorting row (meaning one permutation) wise
    permutations = np.argsort(
        rand.rand(len(data) // lcm_prob_denominators, lcm_prob_denominators),
        axis=1,
    )
    # lookup treatment name for permutations. This works because we flatten
    # row-major style, i.e. one row after another.
    data.loc[:, "treat"] = treat_mask[permutations].flatten(order="C")
    data = data[data["fake"] == 0].drop(columns=["fake"])

    # re-assign type now that we have removed the fake data
    data[idx_col] = data[idx_col].astype(idx_col_type)

    data["treat"] = data["treat"].astype(np.int64)

    return data

# -*- coding: utf-8 -*-
"""
Created on Thursday, 8th November 2018 2:34:47 pm
===============================================================================
@filename:  stochatreat.py
@author:    Manuel Martinez (manmartgarc@gmail.com)
@project:   stochatreat
@purpose:   Define a function that assign treatments over an arbitrary
            number of strata.
===============================================================================
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from stochatreat.utils import get_lcm_prob_denominators


def stochatreat(
    data: pd.DataFrame,
    stratum_cols: List[str],
    treats: int,
    probs: Optional[List[float]] = None,
    random_state: int = 42,
    idx_col: Optional[str] = None,
    size: Optional[int] = None,
    misfit_strategy: str = "stratum",
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
        >>> data = data.merge(treats, how='left', on='myid')

    Multiple strata:
        >>> treats = stochatreat(data=data,
                                 stratum_cols=['stratum1', 'stratum2'],
                                 treats=2,
                                 probs=[1/3, 2/3],
                                 idx_col='myid',
                                 random_state=42)
        >>> data = data.merge(treats, how='left', on='myid')
    """
    # pylint: disable=invalid-name
    R = np.random.RandomState(random_state)

    # =========================================================================
    # do checks
    # =========================================================================
    data = data.copy()

    # create treatment array and probability array
    treatment_ids = list(range(treats))
    # if no probabilities stated
    if probs is None:
        frac = 1 / len(treatment_ids)
        probs_np = np.array([frac] * len(treatment_ids))
    elif probs is not None:
        probs_np = np.array(probs)
        if probs_np.sum() != 1:
            raise ValueError("The probabilities must add up to 1")

    assertmsg = "length of treatments and probs must be the same"
    assert len(treatment_ids) == len(probs_np), assertmsg

    # check if dataframe is empty
    if data.empty:
        raise ValueError("Make sure that your dataframe is not empty.")

    # check length of data
    if len(data) < 2:
        raise ValueError("Make sure your data has enough observations.")

    # if idx_col parameter was not defined.
    if idx_col is None:
        data = data.rename_axis("index", axis="index").reset_index()
        idx_col = "index"
    elif not isinstance(idx_col, str):
        raise TypeError("idx_col has to be a string.")

    # retrieve type to check and re-assign in the end
    idx_col_type = data[idx_col].dtype

    # check for unique identifiers
    if data[idx_col].duplicated(keep=False).sum() > 0:
        raise ValueError("Values in idx_col are not unique.")

    # if size is larger than sample universe
    if size is not None and size > len(data):
        raise ValueError("Size argument is larger than the sample universe.")

    # deal with multiple strata
    if isinstance(stratum_cols, str):
        stratum_cols = [stratum_cols]

    if misfit_strategy not in ("stratum", "global"):
        raise ValueError("the strategy must be one of 'stratum' or 'global'")

    # sort data - useful to preserve correspondence between `idx_col` and
    # assignments
    data = data.sort_values(by=idx_col)

    # combine strata cells - by assigning stratum ids
    data["stratum_id"] = data.groupby(stratum_cols).ngroup()

    # keep only ids and concatenated strata
    data = data[[idx_col] + ["stratum_id"]].copy()

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
            )
        )

        data = data.droplevel(level="stratum_id")

        assert sum(reduced_sizes) == len(data)

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
        misfit_data = (
            data.groupby("stratum_id")
            .apply(
                lambda x: x.sample(
                    n=(x.shape[0] % lcm_prob_denominators),
                    replace=False,
                    random_state=random_state,
                )
            )
            .droplevel(level="stratum_id")
        )
        good_form_data = data.drop(index=misfit_data.index)

        # assign the misfits their own stratum and concatenate
        misfit_data.loc[:, "stratum_id"] = np.Inf
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

    data.loc[:, "fake"] = False
    fake_rep.loc[:, "fake"] = True

    data = pd.concat([data, fake_rep], sort=False).sort_values(by="stratum_id")

    # generate random permutations without loop by generating large number of
    # random values and sorting row (meaning one permutation) wise
    permutations = np.argsort(
        R.rand(len(data) // lcm_prob_denominators, lcm_prob_denominators),
        axis=1,
    )
    # lookup treatment name for permutations. This works because we flatten
    # row-major style, i.e. one row after another.
    data.loc[:, "treat"] = treat_mask[permutations].flatten(order="C")
    data = data[~data["fake"]].drop(columns=["fake"])

    # re-assign type - as it might have changed with the addition of fake data
    data[idx_col] = data[idx_col].astype(idx_col_type)

    data["treat"] = data["treat"].astype(np.int64)

    assert data["treat"].isnull().sum() == 0

    return data

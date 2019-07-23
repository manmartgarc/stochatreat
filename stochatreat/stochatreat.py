# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:34:47 2018

===============================================================================
@author:    Manuel Martinez
@project:   stochatreat
@purpose:   Define a function that assign treatments over an arbitrary
            number of strata.
===============================================================================
"""
from typing import List

import pandas as pd
import numpy as np

from .utils import get_lcm_prob_denominators

# %%===========================================================================
# Main
# =============================================================================


def stochatreat(data: pd.DataFrame,
                stratum_cols: List[str],
                treats: int,
                probs: List[float] = [None],
                random_state: int = 42,
                idx_col: str = None,
                size: int = None,
                misfit_strategy: str = "stratum"
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
    stratum_ids

    Usage
    -----
    Single block:
        >>> treats = stochatreat(data=data,             # your dataframe
                                 stratum_cols='block1', # the strata variable
                                 treats=2,              # including control
                                 idx_col='myid',        # the unique id column
                                 random_state=42)       # seed for rng
        >>> data = data.merge(treats, how='left', on='myid')

    Multiple blocks:
        >>> treats = stochatreat(data=data,
                                 stratum_cols=['block1', 'block2'],
                                 treats=2,
                                 probs=[1/3, 2/3],
                                 idx_col='myid',
                                 random_state=42)
        >>> data = data.merge(treats, how='left', on='myid')
    """
    R = np.random.RandomState(random_state)

    # =========================================================================
    # do checks
    # =========================================================================
    data = data.copy()

    # create treatment array and probability array
    treatment_ids = list(range(treats))
    # if no probabilities stated
    if probs == [None]:
        frac = 1 / len(treatment_ids)
        probs = np.array([frac] * len(treatment_ids))
    elif probs != [None]:
        probs = np.array(probs)
        if probs.sum() != 1:
            raise ValueError('The probabilities must add up to 1')

    assertmsg = 'length of treatments and probs must be the same'
    assert len(treatment_ids) == len(probs), assertmsg

    # check if dataframe is empty
    if data.empty:
        raise ValueError('Make sure that your dataframe is not empty.')

    # check length of data
    if len(data) < 1:
        raise ValueError('Make sure your data has enough observations.')

    # if idx_col parameter was not defined.
    if idx_col is None:
        data = data.reset_index(drop=True)
        idx_col = 'index'
    elif type(idx_col) is not str:
        raise TypeError('idx_col has to be a string.')
    # check for unique identifiers
    elif data[idx_col].duplicated(keep=False).sum() > 0:
        raise ValueError('Values in idx_col are not unique.')

    # if size is larger than sample universe
    if size is not None and size > len(data):
        raise ValueError('Size argument is larger than the sample universe.')

    # deal with multiple strata
    if type(stratum_cols) is str:
        stratum_cols = [stratum_cols]

    if misfit_strategy not in ('stratum', 'global'):
        raise ValueError("the strategy must be one of 'stratum' or 'global'")

    # sort data
    data = data.sort_values(by=idx_col)

    # combine block cells
    data = data[[idx_col] + stratum_cols].copy()
    data['stratum'] = data[stratum_cols].astype(str).sum(axis=1)
    strata = sorted(set(data['stratum']))

    # apply weights to each stratum if sampling is wanted
    if size is not None:
        size = int(size)
        # get sampling weights
        strata_fracs = (data['stratum']
            .value_counts(normalize=True)
            .sort_index()
        )
        reduced_sizes = (strata_fracs * size).round().astype(int).tolist()
        # draw sample
        sample = []
        for i, stratum in enumerate(strata):
            stratum_sample = data[data['stratum'] == stratum].copy()
            # draw sample using fractions
            stratum_sample = stratum_sample.sample(
                n=reduced_sizes[i],
                replace=False,
                random_state=random_state
            )
            sample.append(stratum_sample)
        # concatenate samples from each stratum
        data = pd.concat(sample)

        assert sum(reduced_sizes) == len(data)

    # keep only ids and concatenated strata
    data = data[[idx_col] + ['stratum']]

    # Treatment assignment proceeds in two stages within each stratum:
    # 1. In as far as units can be neatly divided in the proportions given by
    #    prob they are so divided.
    # 2. Any leftovers ("misfits") are dealt with using either of the methods
    #    described in the docstring

    # 1. determine how to divide cleanly as much as possible

    # convert all probs to fractions and get the lowest common multiple of 
    # their denominators
    lcm_prob_denominators = get_lcm_prob_denominators(probs)

    # produce the assignment mask that we will use to achieve perfect 
    # proportions
    treat_mask = np.repeat(
        treatment_ids, (lcm_prob_denominators*probs).astype(int)
    )

    # =========================================================================
    # re-arrange strata
    # =========================================================================

    data_strata = []
    global_misfits = []

    # slice the data into strata
    for i, stratum in enumerate(strata):
        # slize data by stratum
        data_stratum = data.loc[data['stratum'] == stratum].copy()

        # if using the `global` strategy, throw misfits in their own stratum
        if misfit_strategy == "global":
            # get the block size
            stratum_size = data_stratum.shape[0]
            n_misfit = stratum_size % lcm_prob_denominators
            # partition into misfits / non-misfits
            misfit_data = data_stratum.sample(n_misfit)
            data_stratum = data_stratum.drop(index=misfit_data.index)

            global_misfits.append(misfit_data)

        data_strata.append(data_stratum)

    if misfit_strategy == "global":
        if 'misfit_stratum' in strata:
            raise ValueError("""There is already a stratum called 
                'misfit_stratum' in the data.""")

        # throw all misfits into a single stratum, then append to the others
        global_misfits = pd.concat(global_misfits)
        global_misfits['stratum'] = 'misfit_stratum'

        data_strata.append(global_misfits)

    # =========================================================================
    # assign treatments
    # =========================================================================
    for data_stratum in data_strata:
        # set up first-round treatment ids in the desired proportions
        stratum_size = data_stratum.shape[0]
        n_repeat_mask = stratum_size // lcm_prob_denominators
        stratum_treatments = np.repeat(treat_mask, n_repeat_mask)

        # add misfit treatment ids
        n_misfit = stratum_size % lcm_prob_denominators

        if n_misfit > 0:
            misfit_treatments = R.choice(
                treatment_ids,
                size=n_misfit,
                p=probs
            )
            stratum_treatments = np.r_[stratum_treatments, misfit_treatments]

        # shuffle, then assign the treatment ids to the stratum
        np.random.shuffle(stratum_treatments)
        data_stratum['treat'] = stratum_treatments

    # concatenate all strata
    data_with_treatments = pd.concat(data_strata, sort=False)

    # make sure the order is the same as the original data
    data_with_treatments = data_with_treatments.sort_values(by=idx_col)

    # add unique integer ids for the blocks
    data_with_treatments["stratum_id"] = (data_with_treatments
        .groupby(["stratum"])
        .ngroup()
    )
    data_with_treatments = data_with_treatments.drop(columns=["stratum"])

    data_with_treatments['treat'] = (data_with_treatments['treat']
        .astype(np.int64)
    )

    assert len(data_with_treatments) == len(data)
    assert data_with_treatments['treat'].isnull().sum() == 0
    return data_with_treatments

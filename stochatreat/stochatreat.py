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
                block_cols: List[str],
                treats: int,
                probs: List[float] = [None],
                random_state: int = 42,
                idx_col: str = None,
                size: int = None,
                misfit_strategy: str = "stratum"
                ) -> pd.DataFrame:
    """
    Takes a dataframe and an arbitrary number of treatments over an
    arbitrary number of blocks or strata.

    Attempts to return equally sized treatment groups, while randomly
    assigning misfits (left overs from groups not divisible by the number
    of treatments).

    Parameters
    ----------
    data            :   The data that contains unique ids and the
                        stratification columns.
    block_cols      :   The columns in 'data' that you want to stratify over.
    treats          :   The number of treatments you would like to
                        implement, including control.
    probs           :   The assignment probabilities for each of the treatments.
    random_state    :   The seed for the rng instance.
    idx_col         :   The column name that indicates the ids for your data.
    size            :   The size of the sample if you would like to sample
                        from your data.
    misfit_strategy :   The strategy used to assign misfits. Can be one of
                        'stratum' or 'global'.
                        If 'stratum', will assign misfits randomly and
                        independently within each stratum using probs.
                        If 'global', will group all misfits into one stratum and
                        do a full assignment procedure in this new stratum with
                        local random assignments of the misfits in this stratum

    Returns
    -------
    pandas.DataFrame with idx_col, treat (treatment assignments) and block_ids

    Usage
    -----
    Single block:
        >>> treats = stochatreat(data=data,             # your dataframe
                                 block_cols='block1',   # the blocking variable
                                 treats=2,              # including control
                                 idx_col='myid',        # the unique id column
                                 random_state=42)       # seed for rng
        >>> data = data.merge(treats, how='left', on='myid')

    Multiple blocks:
        >>> treats = stochatreat(data=data,
                                 block_cols=['block1', 'block2'],
                                 treats=2,
                                 probs=[1/3, 2/3],
                                 idx_col='myid',
                                 random_state=42)
        >>> data = data.merge(treats, how='left', on='myid')
    """

    # =========================================================================
    # do checks
    # =========================================================================
    data = data.copy()

    # create treatment array and probability array
    ts = list(range(treats))
    # if no probabilities stated
    if probs == [None]:
        frac = 1 / len(ts)
        probs = np.array([frac] * len(ts))
    elif probs != [None]:
        probs = np.array(probs)
        if probs.sum() != 1:
            raise ValueError('The probabilities must add up to 1')

    assertmsg = 'length of treatments and probs must be the same'
    assert len(ts) == len(probs), assertmsg

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

    # deal with multiple blocks
    if type(block_cols) is str:
        block_cols = [block_cols]

    if misfit_strategy not in ('stratum', 'global'):
        raise ValueError("misfit_strategy must be one of 'stratum' or 'global'")

    # sort data
    data = data.sort_values(by=idx_col)

    # combine block cells
    data = data[[idx_col] + block_cols].copy()
    data['block'] = data[block_cols].astype(str).sum(axis=1)
    blocks = sorted(set(data['block']))

    # apply weights to each block if sampling is wanted
    if size is not None:
        size = int(size)
        # get sampling weights
        fracs = data['block'].value_counts(normalize=True).sort_index()
        reduced_sizes = (fracs * size).round().astype(int).tolist()
        # draw sample
        sample = []
        for i, block in enumerate(blocks):
            block_sample = data[data['block'] == block].copy()
            # draw sample using fractions
            block_sample = block_sample.sample(n=reduced_sizes[i],
                                               replace=False,
                                               random_state=random_state)
            sample.append(block_sample)
        # concatenate samples from each block
        data = pd.concat(sample)

        assert sum(reduced_sizes) == len(data)

    # keep only ids and concatenated blocks
    data = data[[idx_col] + ['block']]

    # Treatment assignment proceeds in two stages within each block:
    # 1. In as far as units can be neatly divided in the proportions given by
    #    prob they are so divided.
    # 2. Any leftovers ("misfits") are dealt with using either of the methods
    #    described in the docstring

    # 1. determine how to divide cleanly as much as possible

    # convert all probs to fractions and get the lowest common multiple of their
    # denominators
    lcm_prob_denominators = get_lcm_prob_denominators(probs)

    # produce the assignment mask that we will use to achieve perfect proportions
    treat_mask = np.repeat(ts, (lcm_prob_denominators*probs).astype(int))

    # =========================================================================
    # re-arrange blocks
    # =========================================================================

    if misfit_strategy == 'global':
        # separate the global misfits
        misfit_data = data.groupby('block').apply(
            lambda x: x.sample(
                n=(x.shape[0] % lcm_prob_denominators),
                replace=False,
                random_state=random_state
            )
        ).droplevel(level='block')
        good_form_data = data.drop(index=misfit_data.index)

        # assign the misfits their own block and concatenate
        if 'misfit_block' in blocks:
            raise ValueError("There is already a block called 'misfit_block' in the data.")
        misfit_data.loc[:, 'block'] = 'misfit_block'
        data = pd.concat([good_form_data, misfit_data])

    # =========================================================================
    # assign treatments
    # =========================================================================
    
    # sort by strata first, and assign a long list of permuted treat_mask
    # to deal with misfits, in this case we can add fake rows to make it so everything
    # is divisible and toss them later -> no costly apply inside groups

    # to make sure we have no problems when sorting 
    data.loc[:, 'block'] = data['block'].astype(str)

    # add fake rows for each group so the total number can be divided by num_treatments
    fake = pd.DataFrame({'fake': data.groupby('block').size()}).reset_index()  
    fake.loc[:, 'fake'] = (
        (lcm_prob_denominators - fake['fake'] % lcm_prob_denominators) % lcm_prob_denominators
    )
    fake_rep = pd.DataFrame(fake.values.repeat(fake['fake'], axis=0), columns=fake.columns)

    data.loc[:, 'fake'] = False
    fake_rep.loc[:, 'fake'] = True

    ordered = pd.concat([data, fake_rep], sort=False).sort_values(['block'])

    # generate random permutations without loop by generating large number of
    # random values and sorting row (meaning one permutation) wise
    permutations = np.argsort(
        np.random.rand(len(ordered) // lcm_prob_denominators, lcm_prob_denominators),
        axis=1
    )
    # lookup treatment name for permutations. This works because we flatten
    # row-major style, i.e. one row after another.
    ordered['treat'] = treat_mask[permutations].flatten(order='C')
    ordered = ordered[~ordered['fake']].drop('fake', 1)

    data.loc[:, 'treat'] = ordered['treat']

    # add unique integer ids for the blocks
    data['block_id'] = data.groupby(['block']).ngroup()
    data = data.drop(columns=['block'])

    data['treat'] = data['treat'].astype(np.int64)

    assert data['treat'].isnull().sum() == 0
    
    return data

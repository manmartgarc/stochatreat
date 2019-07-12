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
import pandas as pd
import numpy as np
from typing import List

# %%===========================================================================
# Main
# =============================================================================


def stochatreat(data: pd.DataFrame,
                block_cols: List[str],
                treats: int,
                probs: List[float] = [None],
                random_state: int = 42,
                idx_col: str = None,
                size: int = None) -> pd.Series:
    """
    Takes a dataframe and an arbitrary number of treatments over an
    arbitrary number of clusters or strata.

    Attempts to return equally sized treatment groups, while randomly
    assigning misfits (left overs from groups not divisible by the number
    of treatments).

    Parameters
    ----------
    data        : pandas.code.frame.DataFrame
    block_cols  : string or list of strings
    treats      : int
    probs       : int
    random_state: int
    idx_col     : string
    size        : int

    Returns
    -------
    The function returns a dataframe with the treatment assignment that
    can be merged with the original data frame.

    Usage
    -----
    Single cluster:
        >>> treats = stochatreat(data=data,             # your dataframe
                                 block_cols='cluster1', # the blocking variable
                                 treats=2,              # including control
                                 idx_col='myid',        # the unique id column
                                 random_state=42)       # seed for rng
        >>> data = data.merge(treats, how='left', on='myid')

    Multiple clusters:
        >>> treats = stochatreat(data=data,
                                 block_cols=['cluster1', 'cluster2'],
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
    ts = list(range(treats))
    # if no probabilities stated
    if probs == [None]:
        frac = 1 / len(ts)
        probs = np.array([frac] * len(ts))
    elif probs != [None]:
        probs = np.array(probs)
        assertmsg = 'the probabilities must add up to 1'
        assert probs.sum() == 1, assertmsg

    assertmsg = 'length of treatments and probs must be the same'
    assert len(ts) == len(probs), assertmsg

    # check length of data
    if len(data) < 1:
        raise ValueError('Make sure your data has enough observations.')

    # if idx_col parameter was not defined.
    if idx_col is None:
        data = data.reset_index(drop=True)
        idx_col = 'index'
    elif type(idx_col) is not str:
        raise TypeError('idx_col has to be a string.')

    # if size is larger than sample universe
    if size is not None and size > len(data):
        raise ValueError('Size argument is larger than the sample universe.')

    # check for unique identifiers
    if data[idx_col].duplicated(keep=False).sum() > 0:
        raise ValueError('Values in idx_col are not unique.')

    # deal with multiple clusters
    if type(block_cols) is str:
        block_cols = [block_cols]

    # sort data
    data = data.sort_values(by=idx_col)

    # combine cluster cells
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

    # keep only ids and concatenated clusters
    data = data[[idx_col] + ['block']]

    # =========================================================================
    # assign treatments
    # =========================================================================
    slizes = []
    for i, cluster in enumerate(blocks):
        new_slize = []
        # slize data by cluster
        slize = data.loc[data['block'] == cluster].copy()
        # get the block size
        block_size = slize.shape[0]

        # get the number of "adherents"
        treat_blocks = np.floor(block_size * probs)
        n_belong = int(treat_blocks.sum())
        # get the number of misfits
        n_misfit = int(block_size - n_belong)
        # generate indexes to slice
        locs = treat_blocks.cumsum()

        # if there are no misfits
        if n_misfit == 0:
            # assign random values
            slize['rand'] = R.uniform(size=len(slize))
            # sort by random
            slize = slize.sort_values(by='rand')
            # drop the rand column
            slize = slize.drop(columns='rand')
            # reset index in order to keep original id
            slize = slize.reset_index(drop=True)
            # assign treatment by index
            for i, treat in enumerate(ts):
                if treat == 0:
                    slize.loc[:locs[i], 'treat'] = treat
                else:
                    slize.loc[locs[i - 1]:locs[i], 'treat'] = treat
            new_slize = slize.copy()

        # if there are any misfits
        elif n_misfit > 0:
            # separate groups between adherents and misfits
            adherents = slize.iloc[:n_belong].copy()
            misfits = slize.iloc[n_belong:].copy()

            # assign treatmens on each group
            for aux in [adherents, misfits]:
                # assign random values
                aux['rand'] = R.uniform(size=len(aux))
                # sort by random
                aux = aux.sort_values(by='rand')
                # drop the rand column
                aux = aux.drop(columns='rand')
                # reset index in order to keep original id
                aux = aux.reset_index(drop=True)
                # assign treatment by index
                for i, treat in enumerate(ts):
                    if treat == 0:
                        aux.loc[:locs[i], 'treat'] = treat
                    else:
                        aux.loc[locs[i - 1]:locs[i], 'treat'] = treat
                new_slize.append(aux)
                del aux
            new_slize = pd.concat(new_slize)

        # append blocks together
        slizes.append(new_slize)

    # concatenate all blocks together
    ids_treats = pd.concat(slizes, sort=False)
    # make sure the order is the same as the original data
    ids_treats = ids_treats.sort_values(by=idx_col)
    # map the concatenated blocks to block ids to retrieve the blocks
    # within which randomization was done easily
    ids_treats["block_id"] = ids_treats.groupby(["block"]).ngroup()
    ids_treats = ids_treats.drop(columns="block")
    # reset index
    ids_treats = ids_treats.reset_index(drop=True)
    ids_treats['treat'] = ids_treats['treat'].astype(np.int64)

    assert len(ids_treats) == len(data)
    assert ids_treats['treat'].isnull().sum() == 0
    return ids_treats

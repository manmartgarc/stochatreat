# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:34:47 2018

@author:    Manuel Martinez
@email:     manmartgarc@gmail.com
@purpose:   Define a function that assign treatments over an arbitrary
            number of strata.
"""

import pandas as pd
import numpy as np

# %% Main


def stochatreat(df, block_cols, treats, seed=0, idx_col=None, size=None):
    """
    Takes a dataframe and an arbitrary number of treatments over an
    arbitrary number of clusters or strata.

    Attempts to return equally sized treatment groups, while randomly
    assigning misfits (left overs from groups not divisible by the number
    of treatments).

    Parameters
    ----------
    df: string
        DataFrame of your data.
    block_cols: string or list of strings
        Columns of your DataFrame over wich you wish to stratify over.
    treats:
        Number of treatment cells you wish to use (including control).
    size:
        Target sample size.
    seed:
        Seed for the randomization process, default is 0.
    idx_col: string
        DataFrame column with unique identifiers. If empty, uses the
        DataFrame's index as a unique identifier.

    Returns
    -------

    The function returns a pandas DataFrame object, which can be merged back
    to the original dataframe.

    Usage
    -----
    Single cluster:

    >>> treatments = stochatreat(df, 'clusters', 2, seed=1337, idx_col='myid')
    >>> df = df.merge(treatments, on='myid')

    Multiple clusters:

    >>> treatments = stochatreat(df, ['cluster1', 'cluster2'],
                                2, seed=1337, idx_col='myid')
    >>> df = df.merge(treatments, on='myid')
    """
    np.random.seed(seed)

    n_misfits = []
    
    if len(df) < 1:
        raise ValueError('Make sure your data has enough observations')

    # if idx_col parameter was not defined.
    if idx_col is None:
        df.reset_index(drop=True, inplace=True)
        idx_col = 'index'
    elif type(idx_col) is not str:
        raise TypeError('idx_col has to be a string.')

    # if size is larger than sample universe
    if size is not None and size > len(df):
        raise ValueError('Size argument is larger than the sample universe.')

    # check for unique identifiers
    if df[idx_col].duplicated(keep=False).sum() > 0:
        raise ValueError('Values in idx_col are not unique.')

    # deal with multiple clusters
    if type(block_cols) is str:
        block_cols = [block_cols]

    # combine cluster cells
    df = df[[idx_col] + block_cols].copy()
    df['blocks'] = df[block_cols].astype(str).sum(axis=1)

    # apply weights to each block
    # calculate weights if none were given
    if size is not None:
        size = int(size)
        fracs = (df.groupby('blocks')['blocks'].count() / len(df)).values
        reduced_sizes = np.round(fracs * size).astype(int)

    # keep only ids and concatenated clusters
    df = df[df.columns[~df.columns.isin(block_cols)]]

    slizes = []
    for i, cluster in enumerate(sorted(df['blocks'].unique())):
        treats = int(treats)

        # slize df by cluster
        slize = df.loc[df['blocks'] == cluster].copy()
        slize = slize[[idx_col]]

        # slice the slize
        if size is not None:
            reduced_size = reduced_sizes[i]
            slize['rand'] = np.random.uniform(size=len(slize))
            slize.sort_values(by='rand', ascending=False, inplace=True)
            slize = slize.iloc[:max(1, reduced_size), :]
            slize.drop(columns='rand', inplace=True)

        if len(slize) < treats:
            slize['treat'] = np.random.randint(low=1, high=treats,
                                               size=len(slize))
            slize.reset_index(drop=True, inplace=True)

        # attempt to divide cluster into equal groups depending on treatments
        elif (len(slize) % treats == 0 and
              len(slize) >= treats):  # cluster fits into treatments nicely
            treat_block = int(len(slize) / treats)

            # assign random numbers to each household
            slize['rand'] = np.random.uniform(size=len(slize))

            # order by random numbers
            slize.sort_values(by='rand', ascending=False, inplace=True)
            slize.reset_index(drop=True, inplace=True)

            # assign treatments based on treatment blocks
            for i in range(treats):
                # if first treatment cell
                if i == range(treats)[0]:
                    slize.loc[:treat_block, 'treat'] = i
                # if not the first treatment cell
                else:
                    slize.loc[treat_block * i:treat_block * (i + 1),
                              'treat'] = i

        elif (len(slize) % treats != 0 and
              len(slize >= treats)):  # cluster doesn't fit into treats
            # remove extra and classify as misfits
            new_len = len(slize) - (len(slize) % treats)
            new_slize = slize.iloc[:new_len].copy()

            misfits = slize.iloc[new_len:].copy()
            misfits.reset_index(drop=True, inplace=True)

            treat_block = int(len(new_slize) / treats)

            # assign random numbers to each household
            new_slize['rand'] = np.random.uniform(size=len(new_slize))

            # order by random numbers
            new_slize.sort_values(by='rand', ascending=False, inplace=True)
            new_slize.reset_index(drop=True, inplace=True)

            # assign treatments based on treatment blocks
            for i in range(treats):
                # if first treatment cell
                if i == range(treats)[0]:
                    new_slize.loc[:treat_block, 'treat'] = i
                # if not the first treatment cell
                else:
                    new_slize.loc[treat_block * i:treat_block * (i + 1),
                                  'treat'] = i

            # deal with misfits
            misfits['treat'] = np.random.randint(0, treats,
                                                 size=len(misfits))
            n_misfits.append(len(misfits))

            # un-marginalize misfits :skull:
            slize = pd.concat([new_slize, misfits], sort=False)

        try:
            slize.drop(columns='rand', inplace=True)
        except KeyError:
            pass

        slizes.append(slize)
        ids_treats = pd.concat(slizes)
        ids_treats = ids_treats.reset_index(drop=True)

    return ids_treats

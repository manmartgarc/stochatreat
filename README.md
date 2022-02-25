# Stochatreat

![Main Branch Tests](https://github.com/manmartgarc/stochatreat/actions/workflows/build.yml/badge.svg?branch=main)

## Introduction

This is a Python tool to employ stratified randomization or sampling with uneven numbers in some strata using pandas. Mainly thought with RCTs in mind, it also works for any other scenario in where you would like to randomly allocate treatment within *blocks* or *strata*. The tool also supports having multiple treatments with different probability of assignment within each block or stratum.

## Installation

```bash
pip install stochatreat
```

## Usage

Single cluster:

```python
from stochatreat import stochatreat
import numpy as np
import pandas as pd

# make 1000 households in 5 different neighborhoods.
np.random.seed(42)
df = pd.DataFrame(
      data={'id': list(range(1000)),
      'nhood': np.random.randint(1, 6, size=1000)})

# randomly assign treatments by neighborhoods.
treats = stochatreat(
      data=df,                      # your dataframe
      stratum_cols='nhood',         # the blocking variable
      treats=2,                     # including control
      idx_col='id',                 # the unique id column
      random_state=42,              # random seed
      misfit_strategy='stratum')    # the misfit strategy to use
# merge back with original data
df = df.merge(treats, how='left', on='id')

# check for allocations
df.groupby('nhood')['treat'].value_counts().unstack()

# previous code should return this
treat    0    1
nhood
1      105  105
2       95   95
3       95   95
4      103  103
5      102  102
```

Multiple clusters and treatment probabilities:

```python
from stochatreat import stochatreat
import numpy as np
import pandas as pd

# make 1000 households in 5 different neighborhoods, with a dummy indicator
np.random.seed(42)
df = pd.DataFrame(data={'id': list(range(1000)),
                        'nhood': np.random.randint(1, 6, size=1000),
                        'dummy': np.random.randint(0, 2, size=1000)})

# randomly assign treatments by neighborhoods and dummy status.
treats = stochatreat(data=df,
                     stratum_cols=['nhood', 'dummy'],
                     treats=2,
                     probs=[1/3, 2/3],
                     idx_col='id',
                     random_state=42,
                     misfit_strategy='global')
# merge back with original data
df = df.merge(treats, how='left', on='id')

# check for allocations
df.groupby(['nhood', 'dummy'])['treat'].value_counts().unstack()

# previous code should return this
treat         0   1
nhood dummy
1     0      37  75
      1      33  65
2     0      35  69
      1      29  57
3     0      30  58
      1      34  68
4     0      36  72
      1      32  66
5     0      33  68
      1      35  68
```

## Acknowledgments

- `stochatreat` is totally inspired by [Alvaro Carril's](https://acarril.github.io/) fantastic Stata package: [`randtreat`](https://acarril.github.io/posts/randtreat), which was published in [The Stata Journal](https://www.stata-journal.com/article.html?article=st0490).
- [David McKenzie's](http://blogs.worldbank.org/impactevaluations/tools-of-the-trade-doing-stratified-randomization-with-uneven-numbers-in-some-strata) fantastic post (and blog) about running RCTs for the World Bank.
- [*In Pursuit of Balance: Randomization in Practice in Development Field Experiments.* Bruhn, McKenzie, 2009](https://www.aeaweb.org/articles?id=10.1257/app.1.4.200)

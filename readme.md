# Stochatreat
## Introduction
This is a Python module to employ block randomization using pandas. Mainly thought with RCTs in mind, it also works for any other scenario in where you would like to randomly allocate treatment within *blocks* or *strata*.

## Installation
For now the easiest way (for me) to use this is to copy `stochatreat.py`
into wherever you'd like to use it, and then import it using:
```python
from stochatreat import stochatreat
```

## Usage
Single cluster:
```python
import numpy as np
import pandas as pd
from stochatreat import stochatreat

# make 1000 households in 5 different neighborhoods.
np.random.seed(1337)
df = pd.DataFrame(data={'id': list(range(1000)),
                        'nhood': np.random.randint(1, 6, size=1000)})

# randomly assign treatments by neighborhoods.
df['treat'] = stochatreat(data=df,
                          block_cols='nhood',
                          treats=2,  # including control
                          idx_col='id',
                          seed=1337)

# check for allocations
df.groupby('nhood')['treat'].value_counts().unstack()

# previous code should return this
treat  0.0  1.0
nhood          
1       93   92
2      100  100
3      113  113
4       95   94
5      100  100
```

Multiple clusters:

```python
import numpy as np
import pandas as pd
from stochatreat import stochatreat

# make 1000 households in 5 different neighborhoods, with a dummy indicator
np.random.seed(1337)
df = pd.DataFrame(data={'id': list(range(1000)),
                        'nhood': np.random.randint(1, 6, size=1000),
                        'dummy': np.random.randint(0, 2, size=1000)})

# randomly assign treatments by neighborhoods and dummy status.
df['treat'] = stochatreat(data=df,
                          block_cols=['nhood', 'dummy'],
                          treats=2,  # including control
                          idx_col='id',
                          seed=1337)

# check for allocations
df.groupby(['nhood', 'dummy'])['treat'].value_counts().unstack()

# previous code should return this
treat        0.0  1.0
nhood dummy          
1     0       56   55
      1       37   37
2     0       49   50
      1       51   50
3     0       56   56
      1       57   57
4     0       48   47
      1       47   47
5     0       50   49
      1       51   50
```

## Acknowledgments
- `stochatreat` is totally inspired by [Alvaro Carril's](https://acarril.github.io/) fantastic Stata package: [`randtreat`](https://acarril.github.io/posts/randtreat), which was published in [The Stata Journal](https://www.stata-journal.com/article.html?article=st0490) :trumpet:.
- [David McKenzie's](http://blogs.worldbank.org/impactevaluations/tools-of-the-trade-doing-stratified-randomization-with-uneven-numbers-in-some-strata) fantastic post (and blog) about running RCTs for the World Bank.
- [*In Pursuit of Balance: Randomization in Practice in Development Field Experiments.* Bruhn, McKenzie, 2009](https://www.aeaweb.org/articles?id=10.1257/app.1.4.200)

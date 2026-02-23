# Usage

## Installation

=== ":material-language-python: pip"

    ```bash
    pip install stochatreat
    ```

=== ":material-snake: conda"

    ```bash
    conda install -c conda-forge stochatreat
    ```

## How it works

`stochatreat` assigns treatments within each stratum independently. For a given set of treatment probabilities, it:

1. Divides each stratum into the largest possible block that can be split in exact proportion
2. Assigns treatments to the remainder (*misfits*) using one of three strategies

### Misfit strategies

| Strategy | Behaviour |
|---|---|
| `"stratum"` *(default)* | Misfits in each stratum are assigned randomly and independently using the given probabilities |
| `"global"` | All misfits across strata are pooled into one group and assigned together |
| `"none"` | Misfits are left unassigned (`treat = NA`) and marked with `stratum_id = NA` for manual handling |

## Examples

### Single stratum

```python
from stochatreat import stochatreat
import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.DataFrame(
    data={"id": range(1000), "nhood": np.random.randint(1, 6, size=1000)}
)

treats = stochatreat(
    data=df,
    stratum_cols="nhood",
    treats=2,
    idx_col="id",
    random_state=42,
    misfit_strategy="stratum",
)
df = df.merge(treats, how="left", on="id")
df.groupby("nhood")["treat"].value_counts().unstack()
```

```
treat    0    1
nhood
1      105  105
2       95   95
3       95   95
4      103  103
5      102  102
```

### Multiple strata and unequal probabilities

```python
np.random.seed(42)
df = pd.DataFrame(
    data={
        "id": range(1000),
        "nhood": np.random.randint(1, 6, size=1000),
        "dummy": np.random.randint(0, 2, size=1000),
    }
)

treats = stochatreat(
    data=df,
    stratum_cols=["nhood", "dummy"],
    treats=2,
    probs=[1 / 3, 2 / 3],
    idx_col="id",
    random_state=42,
    misfit_strategy="global",
)
df = df.merge(treats, how="left", on="id")
df.groupby(["nhood", "dummy"])["treat"].value_counts().unstack()
```

```
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

### Sampling from a larger population

Use `size` to draw a stratified sample before assigning treatments:

```python
treats = stochatreat(
    data=df,
    stratum_cols="nhood",
    treats=2,
    idx_col="id",
    size=500,
    random_state=42,
)
```

### Manual misfit handling

The `"none"` strategy identifies misfits but leaves their treatment unassigned. This is useful when you want to handle these cases manually:

```python
# Identify misfits without assigning treatments to them
treats = stochatreat(
    data=df,
    stratum_cols="nhood",
    treats=2,
    idx_col="id",
    random_state=42,
    misfit_strategy="none",
)

# Misfits are marked with stratum_id = NA and treat = NA
misfits = treats[treats["stratum_id"].isna()]
print(f"Found {len(misfits)} misfits")

# Option 1: Assign all misfits to control
treats.loc[treats["stratum_id"].isna(), "treat"] = 0

# Option 2: Exclude misfits from the study
df = df.merge(treats, how="left", on="id")
df_assigned = df[df["treat"].notna()]
```

## References

- `stochatreat` is inspired by [Alvaro Carril's](https://acarril.github.io/) Stata package [`randtreat`](https://acarril.github.io/posts/randtreat), published in [The Stata Journal](https://www.stata-journal.com/article.html?article=st0490).
- [Tools of the trade: Doing Stratified Randomization with Uneven Numbers in some Strata](http://blogs.worldbank.org/impactevaluations/tools-of-the-trade-doing-stratified-randomization-with-uneven-numbers-in-some-strata) on stratified randomization for the World Bank.
- [*In Pursuit of Balance: Randomization in Practice in Development Field Experiments.* Bruhn, McKenzie, 2009](https://www.aeaweb.org/articles?id=10.1257/app.1.4.200)

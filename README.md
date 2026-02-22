# Stochatreat

<p align="center">
  <img src="docs/assets/stochatreat.png" width="200" alt="stochatreat logo">
</p>

<p align="center">
  <a href="https://github.com/manmartgarc/stochatreat/actions/workflows/test.yml"><img src="https://github.com/manmartgarc/stochatreat/actions/workflows/test.yml/badge.svg" alt="typing-and-tests"></a>
  <a href="https://coveralls.io/github/manmartgarc/stochatreat?branch=main"><img src="https://coveralls.io/repos/github/manmartgarc/stochatreat/badge.svg?branch=main" alt="Coverage Status"></a>
  <a href="https://pypi.org/project/stochatreat/"><img src="https://img.shields.io/pypi/v/stochatreat?logo=pypi" alt="pypi"></a>
  <a href="https://pepy.tech/projects/stochatreat"><img src="https://static.pepy.tech/badge/stochatreat/month" alt="PyPI Downloads"></a>
  <a href="https://anaconda.org/conda-forge/stochatreat"><img src="https://img.shields.io/conda/v/conda-forge/stochatreat?logo=conda-forge" alt="Conda"></a>
  <img src="https://img.shields.io/conda/dn/conda-forge/stochatreat?logo=conda-forge" alt="conda-downloads">
  <a href="https://github.com/pypa/hatch"><img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" alt="Hatch project"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" alt="linting - Ruff"></a>
  <a href="https://github.com/astral-sh/ty"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" alt="ty"></a>
  <a href="https://spdx.org/licenses/"><img src="https://img.shields.io/badge/license-MIT-9400d3.svg" alt="License - MIT"></a>
</p>

---

Stratified random treatment assignment using pandas. Designed for randomized controlled trials (RCTs) — assign treatments across any number of strata, with equal or unequal probabilities, and let `stochatreat` handle the misfits.

## Installation

```bash
pip install stochatreat
# or
conda install -c conda-forge stochatreat
```

## Quick start

```python
from stochatreat import stochatreat
import numpy as np
import pandas as pd

df = pd.DataFrame({"id": range(1000), "nhood": np.random.randint(1, 6, 1000)})

treats = stochatreat(data=df, stratum_cols="nhood", treats=2, idx_col="id", random_state=42)
df = df.merge(treats, how="left", on="id")
```

For full documentation and examples visit **[manmartgarc.github.io/stochatreat](https://manmartgarc.github.io/stochatreat/)**.

## Contributing

Read the [contributing guide](https://github.com/manmartgarc/stochatreat/blob/main/CONTRIBUTING.md).

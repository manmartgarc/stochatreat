---
hide:
  - toc
---

# stochatreat :material-dice-multiple:

<figure markdown="span">
  ![stochatreat](assets/stochatreat.png){ width="250" }
</figure>

**Stratified random treatment assignment using pandas.**

---

Got an RCT with unequal strata sizes? `stochatreat` handles the messiness for you: it maximises balance within every stratum and deals with the leftover units (*misfits*) so your treatment groups stay as equal as possible.

!!! tip "Why stochatreat?"
    - :material-check-circle: Works with **any number of strata**
    - :material-check-circle: Supports **unequal treatment probabilities**
    - :material-check-circle: Three misfit strategies: `"stratum"`, `"global"`, or `"none"`
    - :material-check-circle: Reproducible via `random_state`
    - :material-check-circle: Returns a clean pandas DataFrame — just merge and go

## Install

=== ":material-language-python: pip"

    ```bash
    pip install stochatreat
    ```

=== ":material-snake: conda"

    ```bash
    conda install -c conda-forge stochatreat
    ```

[Get started :material-arrow-right:](usage.md){ .md-button .md-button--primary }
[API Reference :material-arrow-right:](api.md){ .md-button }

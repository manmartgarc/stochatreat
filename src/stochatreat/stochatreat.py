"""Stratified random assignment of treatments to units.

This module provides a function to assign treatments to units in a
stratified manner. The function is designed to work with pandas
dataframes and is able to handle multiple strata. There are also different
strategies to deal with misfits (units that are left over after the
stratified assignment procedure).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from stochatreat.misfit import MisfitStrategy, make_misfit_handler
from stochatreat.preparation import DataPreparator
from stochatreat.treatment import TreatmentAssigner, TreatmentSpec


def stochatreat(
    data: pd.DataFrame,
    stratum_cols: list[str] | str,
    treats: int,
    probs: list[float] | None = None,
    random_state: int | None = 42,
    idx_col: str | None = None,
    size: int | None = None,
    misfit_strategy: MisfitStrategy = "stratum",
) -> pd.DataFrame:
    """Assign treatments to units in a stratified manner.

    Takes a dataframe and an arbitrary number of treatments over an
    arbitrary number of strata.

    Attempts to return equally sized treatment groups, while randomly
    assigning misfits (left overs from strata not divisible by the number
    of treatments).

    Args:
        data: The data that contains unique ids and the stratification columns.
        stratum_cols: The columns in 'data' that you want to stratify over.
        treats: The number of treatments you would like to implement,
            including control.
        probs: The assignment probabilities for each of the treatments.
        random_state: The seed for the rng instance.
        idx_col: The column name that indicates the ids for your data.
        size: The size of the sample if you would like to sample from your
            data.
        misfit_strategy: The strategy used to assign misfits. Can be one of
            'stratum' or 'global'. If 'stratum', will assign misfits randomly
            and independently within each stratum using probs. If 'global',
            will group all misfits into one stratum and do a full assignment
            procedure in this new stratum with local random assignments of the
            misfits in this stratum.

    Returns:
        pandas.DataFrame with idx_col, treat (treatment assignments) and
        stratum_id (the id of the stratum within which the assignment
        procedure was carried out) columns.

    Examples:
        Single stratum:

        >>> treats = stochatreat(data=data,               # your dataframe
                                 stratum_cols='stratum1', # stratum variable
                                 treats=2,                # including control
                                 idx_col='myid',          # unique id column
                                 random_state=42)         # seed for rng
        >>> data = data.merge(treats, how="left", on="myid")

        Multiple strata:

        >>> treats = stochatreat(data=data,
                                 stratum_cols=['stratum1', 'stratum2'],
                                 treats=2,
                                 probs=[1/3, 2/3],
                                 idx_col='myid',
                                 random_state=42)
        >>> data = data.merge(treats, how="left", on="myid")

    """
    spec = TreatmentSpec(treats, probs, random_state)
    preparator = DataPreparator(
        data, stratum_cols, idx_col, size, random_state
    )
    prepared_data, resolved_idx_col = preparator.prepare()
    handler = make_misfit_handler(misfit_strategy)
    prepared_data = handler.handle(
        prepared_data, spec.lcm_prob_denominators, random_state
    )
    assigner = TreatmentAssigner(spec)
    return assigner.assign(prepared_data, resolved_idx_col)

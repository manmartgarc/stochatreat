"""Misfit handling strategies for treatment assignment."""

from __future__ import annotations

from typing import Literal, Protocol, get_args

import pandas as pd

MisfitStrategy = Literal["stratum", "global"]


class MisfitHandler(Protocol):
    """Protocol for misfit handling strategies.

    A misfit handler re-arranges strata data before treatment assignment
    so that misfits (units left over when a stratum is not evenly
    divisible by the LCM of probability denominators) are dealt with
    according to the chosen strategy.
    """

    @staticmethod
    def handle(
        data: pd.DataFrame,
        lcm: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Re-arrange strata data according to the misfit strategy.

        Args:
            data: DataFrame with idx_col and stratum_id columns.
            lcm: LCM of the probability denominators.
            random_state: Seed for reproducible sampling.

        Returns:
            Re-arranged DataFrame ready for treatment assignment.

        """


class StratumMisfitHandler(MisfitHandler):
    """Misfit strategy that keeps misfits inside their original stratum.

    Misfits receive a random treatment draw within their own stratum.
    This is the default behaviour.
    """

    @staticmethod
    def handle(
        data: pd.DataFrame,
        lcm: int,  # noqa: ARG004
        random_state: int | None,  # noqa: ARG004
    ) -> pd.DataFrame:
        """Return data unchanged; misfits stay in their original stratum.

        Args:
            data: DataFrame with stratum_id column.
            lcm: LCM of probability denominators (unused here).
            random_state: RNG seed (unused here).

        Returns:
            The input data unmodified.

        """
        return data


class GlobalMisfitHandler(MisfitHandler):
    """Misfit strategy that pools all misfits into a single global stratum.

    Extracts the leftover units from every stratum (those that prevent
    perfect divisibility) and assigns them to a synthetic stratum with
    id ``-1``. The full assignment procedure is then run on this combined
    misfit stratum.
    """

    @staticmethod
    def handle(
        data: pd.DataFrame,
        lcm: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Pool misfits from all strata into stratum -1.

        Args:
            data: DataFrame with stratum_id column.
            lcm: LCM of probability denominators used to determine
                how many units are misfits per stratum.
            random_state: Seed for reproducible misfit sampling.

        Returns:
            DataFrame where misfits have been moved to stratum -1.

        """
        misfit_data = data.groupby("stratum_id").apply(
            lambda x: x.sample(
                n=(x.shape[0] % lcm),
                replace=False,
                random_state=random_state,
            ),
            include_groups=False,
        )
        misfit_data["stratum_id"] = misfit_data.index.get_level_values(0)
        misfit_data = misfit_data.droplevel(level="stratum_id")
        good_form_data = data.drop(index=misfit_data.index)
        misfit_data.loc[:, "stratum_id"] = -1
        return pd.concat([good_form_data, misfit_data])


def make_misfit_handler(strategy: MisfitStrategy) -> MisfitHandler:
    """Return the appropriate misfit handler for the given strategy.

    Args:
        strategy: One of ``'stratum'`` or ``'global'``.

    Returns:
        A misfit handler instance.

    Raises:
        ValueError: If strategy is not a valid MisfitStrategy.

    """
    valid = get_args(MisfitStrategy)
    if strategy not in valid:
        msg = f"misfit_strategy must be one of {valid}."
        raise ValueError(msg)
    if strategy == "global":
        return GlobalMisfitHandler()
    return StratumMisfitHandler()

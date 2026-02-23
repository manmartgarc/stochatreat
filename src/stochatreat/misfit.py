"""Misfit handling strategies for treatment assignment."""

from __future__ import annotations

from typing import Literal, Protocol, get_args

import numpy as np
import pandas as pd

MisfitStrategy = Literal["stratum", "global", "none"]


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


def _extract_misfits(
    data: pd.DataFrame, lcm: int, random_state: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract misfits from each stratum.

    Args:
        data: DataFrame with stratum_id column.
        lcm: LCM of probability denominators.
        random_state: Seed for reproducible sampling.

    Returns:
        Tuple of (non-misfit data, misfit data).

    """
    rng = np.random.RandomState(random_state)
    data = data.copy()
    data["_random"] = rng.rand(len(data))
    data = data.sort_values(["stratum_id", "_random"])

    group_ranks = data.groupby("stratum_id").cumcount()
    group_sizes = data.groupby("stratum_id")["stratum_id"].transform("count")
    is_misfit = group_ranks < (group_sizes % lcm)

    misfit_data = data[is_misfit].drop(columns=["_random"])
    good_form_data = data[~is_misfit].drop(columns=["_random"])

    return good_form_data, misfit_data


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
        good_form_data, misfit_data = _extract_misfits(data, lcm, random_state)
        misfit_data.loc[:, "stratum_id"] = -1
        return pd.concat([good_form_data, misfit_data])


class NoneMisfitHandler(MisfitHandler):
    """Misfit strategy that leaves misfits unassigned.

    Identifies misfits and marks them with stratum_id = NA, leaving
    their treatment assignment as NaN. This allows users to identify
    and handle misfits manually.
    """

    @staticmethod
    def handle(
        data: pd.DataFrame,
        lcm: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Mark misfits with stratum_id = NA for later exclusion.

        Args:
            data: DataFrame with stratum_id column.
            lcm: LCM of probability denominators used to determine
                how many units are misfits per stratum.
            random_state: Seed for reproducible misfit sampling.

        Returns:
            DataFrame where misfits have stratum_id = NA.

        """
        good_form_data, misfit_data = _extract_misfits(data, lcm, random_state)
        data = pd.concat([good_form_data, misfit_data])
        data["stratum_id"] = data["stratum_id"].astype("Int64")
        data.loc[misfit_data.index, "stratum_id"] = pd.NA
        return data


def make_misfit_handler(strategy: MisfitStrategy) -> MisfitHandler:
    """Return the appropriate misfit handler for the given strategy.

    Args:
        strategy: One of ``'stratum'``, ``'global'``, or ``'none'``.

    Returns:
        A misfit handler instance.

    Raises:
        ValueError: If strategy is not a valid MisfitStrategy.

    """
    handlers: dict[MisfitStrategy, MisfitHandler] = {
        "stratum": StratumMisfitHandler(),
        "global": GlobalMisfitHandler(),
        "none": NoneMisfitHandler(),
    }
    if strategy not in handlers:
        valid = get_args(MisfitStrategy)
        msg = f"misfit_strategy must be one of {valid}."
        raise ValueError(msg)
    return handlers[strategy]

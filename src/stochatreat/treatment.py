"""Treatment specification and assignment logic."""

from __future__ import annotations

import math
from fractions import Fraction
from math import lcm as _lcm
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable


def get_lcm_prob_denominators(probs: Iterable[float]) -> int:
    """Compute the LCM of the denominators of the probabilities.

    Args:
        probs: An iterable of probability values.

    Returns:
        The least common multiple of the denominators.

    """
    prob_denominators = (
        Fraction(prob).limit_denominator().denominator for prob in probs
    )
    return _lcm(*prob_denominators)


class TreatmentSpec:
    """Holds and validates treatment configuration.

    Encapsulates the number of treatments, assignment probabilities,
    and the random number generator. Computes derived values such as
    the LCM of probability denominators and the treatment mask used
    during assignment.

    Args:
        treats: Number of treatments, including control.
        probs: Assignment probability for each treatment. If None,
            equal probabilities are used.
        random_state: Seed for the random number generator.

    Raises:
        ValueError: If probabilities do not sum to 1 or if the number
            of probabilities does not match the number of treatments.

    """

    def __init__(
        self,
        treats: int,
        probs: list[float] | None,
        random_state: int | None,
    ) -> None:
        """Validate and compute treatment configuration from inputs.

        Raises:
            ValueError: See class docstring.

        """
        self.treatment_ids: list[int] = list(range(treats))
        if probs is None:
            frac = 1 / len(self.treatment_ids)
            self.probs: np.ndarray = np.array([frac] * len(self.treatment_ids))
        else:
            self.probs = np.array(probs)
            if not math.isclose(self.probs.sum(), 1, rel_tol=1e-9):
                msg = "The probabilities must add up to 1"
                raise ValueError(msg)
            if len(self.probs) != len(self.treatment_ids):
                msg = (
                    "The number of probabilities must match the number of "
                    "treatments"
                )
                raise ValueError(msg)
        self.rng: np.random.RandomState = np.random.RandomState(random_state)
        self.lcm_prob_denominators: int = get_lcm_prob_denominators(self.probs)
        self.treat_mask: np.ndarray = np.repeat(
            self.treatment_ids,
            (self.lcm_prob_denominators * self.probs).round().astype(int),
        )


class TreatmentAssigner:
    """Assigns treatments to units using a vectorised permutation approach.

    Adds fake rows so every stratum is divisible by the LCM of
    probability denominators, generates random permutations, maps them
    through the treatment mask, then discards the fake rows and
    restores the original idx_col dtype.

    Args:
        spec: A fully configured TreatmentSpec instance.

    """

    def __init__(self, spec: TreatmentSpec) -> None:
        """Store the TreatmentSpec used during assignment."""
        self._spec = spec

    def assign(self, data: pd.DataFrame, idx_col: str) -> pd.DataFrame:
        """Run the permutation-based assignment and return results.

        Args:
            data: Prepared DataFrame with idx_col and stratum_id columns.
            idx_col: Name of the unique-id column.

        Returns:
            DataFrame with idx_col, stratum_id, and treat columns.

        """
        spec = self._spec
        lcm = spec.lcm_prob_denominators

        idx_col_type = data[idx_col].dtype

        fake = pd.DataFrame({"fake": data.groupby("stratum_id").size()})
        fake = fake.reset_index()
        fake.loc[:, "fake"] = (lcm - fake["fake"] % lcm) % lcm
        fake_rep = pd.DataFrame(
            fake.values.repeat(fake["fake"], axis=0), columns=fake.columns
        )
        # Protect idx_col from silent float upcast when NaNs are introduced
        # by the fake rows; the original dtype is restored after removal.
        data[idx_col] = data[idx_col].astype(object)
        data.loc[:, "fake"] = 0
        fake_rep.loc[:, "fake"] = 1

        data = pd.concat([data, fake_rep], sort=False).sort_values(
            by="stratum_id", kind="stable"
        )

        permutations = np.argsort(
            spec.rng.rand(len(data) // lcm, lcm),
            axis=1,
        )
        data.loc[:, "treat"] = spec.treat_mask[permutations].flatten(order="C")
        data = data[data["fake"] == 0].drop(columns=["fake"])

        data[idx_col] = data[idx_col].astype(idx_col_type)
        data["treat"] = data["treat"].astype(np.int64)

        return data

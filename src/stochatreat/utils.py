"""Utilities for stochatreat."""

from collections.abc import Iterable
from fractions import Fraction
from math import lcm


def get_lcm_prob_denominators(probs: Iterable[float]) -> int:
    """Compute the LCM of the denominators of the probabilities.

    Returns:
        The least common multiple of the denominators.

    """
    prob_denominators = (
        Fraction(prob).limit_denominator().denominator for prob in probs
    )
    return lcm(*prob_denominators)

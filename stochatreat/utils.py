from fractions import Fraction
from math import lcm
from typing import Iterable


def get_lcm_prob_denominators(probs: Iterable[float]):
    """
    Helper function to compute the LCM of the denominators of the probabilities
    """
    prob_denominators = [
        Fraction(prob).limit_denominator().denominator for prob in probs
    ]
    lcm_prob_denominators = lcm(*prob_denominators)
    return lcm_prob_denominators

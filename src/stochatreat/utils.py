import sys
from fractions import Fraction
from typing import Iterable

if sum(sys.version_info[:2]) < 12:
    from functools import reduce
    from math import gcd  # type: ignore

    def lcm(*args):
        """
        Helper function to compute the Lowest Common Multiple of a list of
        integers
        """
        return reduce(lambda a, b: a * b // gcd(a, b), args)

else:
    from math import lcm  # type: ignore


def get_lcm_prob_denominators(probs: Iterable[float]):
    """
    Helper function to compute the LCM of the denominators of the probabilities
    """
    prob_denominators = [
        Fraction(prob).limit_denominator().denominator for prob in probs
    ]
    lcm_prob_denominators = lcm(*prob_denominators)
    return lcm_prob_denominators

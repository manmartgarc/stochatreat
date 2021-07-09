from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Iterable


def lcm(ints: Iterable[int]):
    """
    Helper function to compute the Lowest Common Multiple of a list of integers
    """
    return reduce(lambda a, b: a * b // gcd(a, b), ints)


def get_lcm_prob_denominators(probs: Iterable[float]):
    """
    Helper function to compute the LCM of the denominators of the probabilities
    """
    prob_denominators = [
        Fraction(prob).limit_denominator().denominator for prob in probs
    ]
    lcm_prob_denominators = lcm(prob_denominators)
    return lcm_prob_denominators

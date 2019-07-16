# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:15:59 2019

===============================================================================
@author:    Manuel Martinez
@project:   stochatreat
@purpose:   Calculate possible ways to assign misfits to treatment using
            combinatronics.
@inputs:
@outputs:
===============================================================================
"""
from scipy.special import comb


def treat_combs(treats: int, misfits: int) -> float:
    """
    Calculating the number of treatment assignment combinations for the
    remaining misfits.

    Taken from
    http://blogs.worldbank.org/impactevaluations/
    tools-of-the-trade-doing-stratified-randomization
    -with-uneven-numbers-in-some-strata
    """
    return comb(treats, misfits)

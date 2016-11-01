# -*- coding: utf-8 -*-
import numpy as np

def Weibull_S_Shaped(x,a,b,c):
    """
    Parameters
    ----------
    X : array-like
        function arguments
    a : number
        expected cumulative total number of defects
        a > 0
    b : number
        defect detection rate
        b > 0
    c : number
        defect detection rate booster
        c > 0
    Returns
    -------
    return : array-like
        Returns result of applying function to x with parameters a,b,c
    """
    return a * (1 - (1 + b * (x**c)) * np.exp(-b*(x**c)))

# -*- coding: utf-8 -*-

def Weibull_S_Shaped(x,a,b,c):
    """
    """
    return a * (1 - (1 + b * (x**c)) * np.exp(-b*(x**c)))

# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit

def GoF(y_predicted, y, param_number):
    """
    Goodness of Fit shows how well the model fits the original data. 
    GoF is calculated as a sum of squared residuals divided by the number of degrees of freedom
    
    :param y_predicted: NumPy array of predicted values
    :param y: NumPy array of actual values
    :param param_number: number of model parametres. Used to estimate degrees of freedom.
    :return: float number
    """
    return np.sum(np.power(y_predicted - y, 2)) / (len(y) - param_number)

def PA(trainset, fit_func, x_label='x', y_label='y'):
    """
    Predictive Ability shows how early in the testing 
    the model is able to predict the final number 
    of defects with maximum 10% error. Measured in percentages.
    """
    last = len(trainset)
    for i in range(1, last-1):
        tr = trainset.head(last - i)
        popt, pcov = curve_fit(fit_func, tr[x_label].values, tr[y_label].values)
        a = popt[0]
        error = AcFP(trainset.y.iloc[-1], a)
        if error > 10:
            pa = round(1 - (i-1)/last, 6)
            break
    return pa*100

def AcFP(real, predicted):
    """
    The accuracy of final point (AcFP) corresponds to the effectiveness of the model 
    in determining the final number of defects observed in the dataset.
    """
    return np.abs(real - predicted) / real * 100



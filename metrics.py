# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit
from .sklearn_model import SoftwareReliabilityGrowthModel as SRGM

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

def PA(trainset, fit_func, x_label='x', y_label='y', error_threshold=10):
    """
    Predictive Ability shows how early in the testing 
    the model is able to predict the final number 
    of defects with maximum 10% error. Measured in percentages.
    """
    last = len(trainset)
    for i in range(1, last-1):
        tr = trainset.head(last - i)
        model = SRGM(fit_func)
        model.fit(tr[x_label].values, tr[y_label].values)
        if model.popt is not None:
            a = model.popt[0]
            error = AcFP(trainset.y.iloc[-1], a)
            if error > error_threshold:
                pa = round(1 - (i-1)/last, 6)
                mode = 'normal'
                break
        else:
            pa = round(1 - (i-1)/last, 6)
            mode = 'fitting problem'
            break
    return pa*100, mode

def AcFP(real, predicted):
    """
    The accuracy of final point (AcFP) corresponds to the effectiveness of the model 
    in determining the final number of defects observed in the dataset.
    """
    return np.abs(real - predicted) / real * 100



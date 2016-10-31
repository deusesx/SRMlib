# -*- coding: utf-8 -*-

def GoF(y_predicted, y, param_number):
    """
    Goodness of Fit shows how well the model fits the original data. GoF is calculated as a sum of squared residuals divided by the number of degrees of freedom
    :param y_predicted: NumPy array of predicted values
    :param y: NumPy array of actual values
    :param param_number: number of model parametres. Used to estimate degrees of freedom.
    :return: float number
    """
    return np.sum(np.power(y_predicted - y, 2)) / (len(y) - param_number)

def PA(trainset):
    """
    Predictive Ability shows how early in the testing 
    the model is able to predict the final number 
    of defects with maximum 10% error. Measured in percentages.
    """
    last = len(trainset)
    for i in range(1, last-1):
        tr = trainset.head(last - i)
        popt, pcov = curve_fit(Weibull_S_Shaped, tr.x.values, tr.y.values)
        a = popt[0]
        error = AcFP(trainset.y.iloc[-1], a)
        if error > 10:
            pa = round(1 - (i-1)/last, 6)
            break
    return(pa*100, error)

def AcFP(real, predicted):
    return np.abs(real - predicted) / real * 100



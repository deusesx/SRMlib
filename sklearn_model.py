import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import curve_fit


class SoftwareReliabilityGrowthModel(BaseEstimator, RegressorMixin):
    def __init__(self, approx_func):
        """
        Parameters
        ----------
        approx_func : function
            The function that used for software reliability growth modeling
        """
        self.approx_func = approx_func
    
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values.
        Returns
        -------
        self : object
            Returns object with fitted model parameters in popt
        """
        if len(X.shape) == 1: 
            X = X.reshape(-1, 1)
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.popt, self.pcov = curve_fit(self.approx_func, X.flatten(), y)
        return self
    
    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns prediction for X
        """
        if len(X.shape) == 1: 
            X = X.reshape(-1, 1)
        X = check_array(X)
        return self.approx_func(X.flatten(), *self.popt)


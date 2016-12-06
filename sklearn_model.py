# model libraries
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import curve_fit
import copy

import pickle

from SRMlib.models import Weibull_S_Shaped

def abs_accuracy(a, b):
    return np.abs(a - b) / a * 100

class SoftwareReliabilityGrowthModel(BaseEstimator, RegressorMixin):
    def __init__(self, approx_func='weibull-s'):
        """
        Parameters
        ----------
        approx_func : function
            The function that used for software reliability growth modeling
        """
        models = {'weibull-s': Weibull_S_Shaped}
            
        if type(approx_func) == type("x"):
            self.approx_func = models.get(approx_func)
            self.approx_func_name = approx_func
        else:
            self.approx_func = approx_func
        
        self.fitted = False
        self.popt = None
        self.pcov = None
        
    
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
            X = X.values.reshape(-1, 1)
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        try:
            self.popt, self.pcov = curve_fit(self.approx_func, X.flatten(), y)
            self.fitted = True
        except RuntimeError:
            self.popt = None
            self.pcov = None
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
        if self.fitted:
            if len(X.shape) == 1 and type(X) != np.ndarray: 
                X = X.values.reshape(-1, 1)
            X = check_array(X)
            return self.approx_func(X.flatten(), *self.popt)
        else:
            return None
    
    
    def goodness_of_fit(self, X, y):
        """
        Goodness of Fit shows how well the model fits the original data. 
        GoF is calculated as a sum of squared residuals divided by the number of degrees of freedom
        
        Parameters
        ----------
        X : array-like
            The input samples
        y : array-like
            Ground true values
        Returns
        -------
        GoF : float or None
            Returns goodness of fit 
        """
        if self.fitted:
            if len(X.shape) == 1: 
                X = X.values.reshape(-1, 1)
            y_predicted = self.predict(X)
            param_number = self.approx_func.__code__.co_argcount - 1
            return np.sum(np.power(y_predicted - y, 2)) / (len(y) - param_number)
        return None
    

    def predictive_ability(self, X=None, y=None, error_threshold=10):
        """
        Predictive Ability shows how early in the testing 
        the model is able to predict the final number 
        of defects with maximum 10% error. Measured in percentages.
        
        Parameters
        ----------
        X : array-like
            
        y : array-like
        
        error_threshold : int
            Maximum permissible error in percents
            
        Returns
        -------
        pa : int
            Predictive ability in percents
        mode : String
            Returns predictive ability 
        """
        if not ((X is None) | (y is None)):
            X = self.X_
            y = self.y_
        last = len(y) - 1
        y_last = y[last]
        
        for i in range(1, last):
            model = self.copy()
            model.fit(X[:last - i], y[:last-i])
            if model.popt is not None:
                error = model.accuracy_of_final_point(last, y_last)
                if error > error_threshold:
                    pa = round(1 - (i-1)/last, 6)
                    mode = 'normal'
                    break
            else:
                pa = round(1 - (i-1)/last, 6)
                mode = 'fitting problem'
                break
        return pa*100, mode
    
    
    def accuracy_of_final_point(self, X_final, y_final):
        """
        The accuracy of final point (AcFP) corresponds to the effectiveness of the model 
        in determining the final number of defects observed in the dataset.
        
        Parameters
        ----------
        X_final : int
            
        y_final : int
            
        Returns
        -------
        AcFP : int or None
            Accuracy of final point in percents
        """
        if self.fitted:
            y_final_hat = self.predict(np.array([X_final]))
            return abs_accuracy(y_final, y_final_hat)[0]
        else:
            return None
        
        
    def dump(self, filename):
        """
        Parameters
        ----------
        filename : String
            path to dumping file
        """
        with open(filename, 'wb+') as f:
            pickle.dump(obj=self, file=f)

            
    def load(self, filename):
        """
        Parameters
        ----------
        filename : String
            path to loading file
        """
        with open(filename, 'rb+') as f:
            self = pickle.load(f)
    
    
    def copy(self):
        return copy.deepcopy(self)

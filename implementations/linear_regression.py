### linear_regression.py

import numpy as np


class LinearRegression(object):
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
        self.learning_rate: float = learning_rate
        self.n_iters: int = n_iters
        self.weights: np.ndarray = None
        self.bias: np.ndarray = None
    
    def fit(self, X: np.ndarray, y:np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        
        #init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
    
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
        
# How do you expand this for multilinear regression?
# Worth looking into it
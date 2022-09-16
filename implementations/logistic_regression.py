### logistic_regression.py

from typing import List

import numpy as np


def sigmoid(x) -> np.ndarray:
    return 1/(1 + np.exp(-x))

class LogisticRegression(object):
    
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
        
        self.learning_rate: float = learning_rate
        self.n_iters: int = n_iters
        self.weights: np.ndarray = None
        self.bias: np.ndarray = None
        
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.zeros(n_samples)
        
        for _ in range(self.n_iters):
            linear_pred: np.ndarray = np.dot(X, self.weights) + self.bias
            predictions: np.ndarray = sigmoid(linear_pred)
            
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
    
    def predict(self, X) -> List[float]:
        linear_pred: np.ndarray = np.dot(X, self.weights) + self.bias
        y_pred: np.ndarray = sigmoid(linear_pred)
        
        class_predictions: List[float] = [0 if y <= 0.5 else 1 for y in y_pred]
        
        return class_predictions
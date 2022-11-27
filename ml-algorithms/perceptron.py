### perceptron.py

import numpy as np

def unit_step_func(x):
    return np.where(x >= 0, 1, 0)

class Perceptron(object):
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
        self.learning_rate: float = learning_rate
        self.n_iters: int = n_iters
        self.weights: np.ndarray = None
        self.bias: np.ndarray = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # make sure we have y has labels -1, 1, not 0, 1
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # predict values
                linear_output: np.ndarray = np.dot(x_i, self.weights) + self.bias
                y_pred: int = unit_step_func(linear_output)
                
                # Preceptron update rule:
                # -1 (y_pred is too high) -> decrease weights
                # 1 (y_pred is too low) -> increase weights
                # 0 (correct) -> no change
                
                update = y[idx] - y_pred
                
                self.weights = self.weights + update * x_i
                self.bias = self.bias + self.learning_rate * update
                
    def predict(self, X: np.ndarray):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted:int = unit_step_func(linear_output)
        
        return y_predicted
        
### svm.py

### SVM tries to find the decision boundary (hyperplane) that best separates 2 classes
### The best hyperplane is the one that yields the largest separation between two classes
### i.e. the distance to the nearest data point on each side is maximized


import numpy as np

class SVM(object):
    def __init__(self, learning_rate: float = 0.001, lamda_param: float = 0.01, n_iters: int = 1000):
        self.learning_rate: float = learning_rate
        self.lamda_param: float = lamda_param
        self.n_iters: int = n_iters
        self.weights: np.ndarray = None
        self.bias: int = 0
    
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        
        # make sure we have y has labels -1, 1, not 0, 1
        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        
        # Gradient descent: Minimize J
        # J = (1/n * sum(max(0, 1-y(wx+b)))) + lambda||w||^2
        # J = Hinge loss + Regularization
        # -> Minimize Hinge loss and at the same time maximize the margin
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.b)
                
                # dJ/dw and dJ/db leads to the following update formula
                if condition:
                    self.weights = self.weights - self.learning_rate * (2 * self.lamda_param * self.weights)
                else:
                    self.weights = self.learning_rate * (
                        2 * self.lamda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.b = self.b - self.learning_rate * y_[idx]
        
    def predict(self, X) -> np.ndarray:
        # apply linear model
        approx: np.ndarray = np.dot(X, self.weights) - self.bias
        # use sign to set it to -1 or 1, depending on which side
        # of the decision boundary it is lying
        return np.sign(approx)
### knn.py
### Implementation of k-NN algorithm.

from collections import Counter

import numpy as np
from scipy.stats import mode


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
        params
        ------
        x1 (np.ndarray) of dimensions (N, D)
        x2 (np.ndarray) of dimensions (N', D)
        
        return
        ------
        array of pairwise distances of dimensions (N, N')
    """
    
    # return np.linalg.norm(x1[: np.newaxis] - x2, axis=1)
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


class KNN(object):
    
    def __init__(self, k: int=3):
        self.k = k
    
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = [euclidean_distance(X, x_train) for x_train in self.X]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
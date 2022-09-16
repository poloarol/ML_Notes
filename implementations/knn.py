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
    
    return np.linalg.norm(x1[: np.newaxis] - x2, axis=1)


class KNN(object):
    
    def __init__(self, k: int=3):
        self.k = k
    
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = euclidean_distance(X, self.X)
        topk_idxs = np.argmax(distances, axis=1)[:, :self.k]
        topk_labels = self.y[topk_idxs]
        
        return mode(topk_labels, axis=1)
        
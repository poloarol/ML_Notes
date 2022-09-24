### random_forest.py

import numpy as np
from scipy.stats import mode
from collections import Counter

from typing import List
from decision_tree import DecisionTree

#### Training
# 1. Get a subset of the training data
# 2. Create a decision tree
# 3. Repeat the creation of decision trees several times

#### Testing
# 1. Get the predictions from each tree
# 2. Classification: hold a majority vote
# 3. Regression: get the mean of the predictions


class RandomForest(object):
    def __init__(self, num_trees: int = 10, max_depth: int = 10, min_samples_split: int = 2, num_features: int = None):
        self._num_trees: int = num_trees
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.num_features: int = num_features
        self.trees: List = []
        
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for _ in range(self._num_trees):
            tree = DecisionTree(max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    n_features=self.num_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            
    def _bootstrap_samples(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True) 
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        
        return predictions
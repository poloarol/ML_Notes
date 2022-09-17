### decision_tree.py

import numpy as np
from scipy.stats import mode

class Node(object):
    def __init__(self, feature = None, threshold = None, left = None, right = None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    
    def is_leaf_node(self):
        return self.value is not None
        

class DecisionTree(object):
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, n_features: int = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features))
        self.root = self.grow_tree(X, y)
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> None:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        ### check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_value_label(y)
            return Node(value=leaf_value)
        
        feat_idx = np.random.choice(n_features, self.n_features, replace=False)
        
        ### find the best split
        best_features, best_threshold = self._best_split(X, y, feat_idx)
        
        ### create child node
        left_idxs, right_idxs = self._split(X[:, best_features], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs, :], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs, :], depth+1)
        
        return Node(best_features, best_threshold, left, right)
        
    def _most_common_value_label(self, y) -> int:
        return mode(y, axis=1)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feature_indices:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # calculate the information gain
                gain = self._infromation_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
                    
        return split_idx, split_threshold

    def _information_gain(self, y: np.ndarray, X: np.ndarray, threshold: float) -> float:
        # parent entropy
        parent_entropy = self._entropy(y)
        
        # create children
        left_idx, right_idx = self._split(X, threshold)
        
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        # calculate the weighted entropy of children
        n = len(y)
        num_left, num_right = len(left_idx), len(right_idx)
        entropy_left, entropy_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (num_left/n) * entropy_left + (num_right/n) * entropy_right
        
        # calculate the IG
        information_gain = parent_entropy - child_entropy
        
        return information_gain
        
    def _split(self, X: np.ndarray, threshold: float):
        left_idxs = np.argwhere(X <= threshold).flatten()
        right_idxs = np.argwhere(X > threshold).flatten()
        
        return left_idxs, right_idxs
        
    def _entropy(self, y: np.ndarray) -> float:
        hist = np.bincount(y)
        ps = hist / len(y)
        
        return -np.sum([p * np.log(p) for p in ps if p > 0])
            
    def predict(self, X: np.ndarray):
        return [self._traverse_tree(x) for x in X]
    
    def _traverse_tree(self, x: np.ndarray, node: Node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
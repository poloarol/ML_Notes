### pca.py

import numpy as np

class PCA(object):
    def __init__(self, n_components: int):
        self.n_components: int = n_components
        self.components = None
        self.mean: np.ndarray = None
        
    def predict(self, X):
        # Mean clustering
        self.mean: np.ndarray = np.mean(X, axis=0)
        X = X - self.mean
        
        # covariance, function needs samples as columns
        cov: np.ndarray = np.cov(X.T)
        
        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # -> eigenvector is column vector: c = [:, i]
        # transpose for easier calculations
        eigenvectors = eigenvectors.T
        # sort eigenvectors
        idxs = np.argsort(eigenvectors)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # store first N eigenvectors
        self.components = eigenvectors[0:self.n_components]
    
    def transform(self, X: np.ndarray):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)
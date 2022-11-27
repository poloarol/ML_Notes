### naive_bayes.py
### Training
# 1. Calculate the mean, variance, and prior (frequency) for each class
### Predictions
# 1. Calculate posterior probabilities for each class
# 2. Choose class with the highest posterior probability

import numpy as np

class NaiveBayes(object):
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        # calculate mean, variance, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes)
        
        for idx, n_class in enumerate(self._classes):
            X_class = X[y==n_class]
            
            self._mean[idx: ] = X_class.mean(axis=0)
            self._var[idx: ] = X_class.var(axis=0)
            self._priors[idx] = X_class.shape[0] / float(n_samples)
    
    def predict(self, X: np.ndarray):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        # calculate posterior probability for each class
        for idx, n_class in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            
            posteriors.append(posterior)
            
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, idx, x):
        mean = self._mean[idx]
        var = self._var[idx]
        
        numerator = np.exp(-((x-mean)**2)/ (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        
        return numerator/denominator

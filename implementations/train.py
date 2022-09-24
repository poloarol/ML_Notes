### test.py

from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from knn import KNN
from pca import PCA
from decision_tree import DecisionTree
from random_forest import RandomForest
from perceptron import Perceptron
from svm import SVM

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


classifiers = {
    "knn": KNN(k=3),
    "tree": DecisionTree(),
    "forest": RandomForest(),
    "logit": LogisticRegression(),
    "svm": SVM(),
    "neuron": Perceptron(),
}

for key, classifier in classifiers.items():
    try:
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        
        accuracy = (np.sum(predictions == y_test)) / len(y_test)
        
        print(f"{key}: Accuracy -> {accuracy: .3f}")
    except Exception as e:
        print(e.__repr__())
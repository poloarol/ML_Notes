### test.py

from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from knn import KNN
from pca import PCA
from decision_tree import DecisionTree
from random_forest import RandomForest
from perceptron import Perceptron
from svm import SVM
from naive_bayes import NaiveBayes

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=1234)


classifiers = {
    "knn": KNN(k=3),
    "tree": DecisionTree(),
    "forest": RandomForest(),
    "logit": LogisticRegression(),
    "svm": SVM(),
    "neuron": Perceptron(),
    "bayes": NaiveBayes(),
}

for key, classifier in classifiers.items():
    try:
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        
        accuracy = (np.sum(y_true == predictions)) / len(y_true)
        
        print(f"{key}: Accuracy -> {accuracy: .3f}")
    except Exception as e:
        print(e.__repr__())
        
reg = LinearRegression(learning_rate=0.01)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_true, predictions)
print(f"Linear Regression: Mean Squared Error {mse: .3f}")
from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np


def main():
    iris = load_iris()
    test_idx = [0, 50, 100]

    # training Data
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)

    # testing data
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    # Train Classifier
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)

    print(clf.predict(test_data))


# Run main
main()

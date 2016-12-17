from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score


def main():

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

    clrTree = tree.DecisionTreeClassifier()
    clrTree = clrTree.fit(x_train, y_train)
    outTree = clrTree.predict(x_test)

    clrKN = KNeighborsClassifier()
    clrKN = clrKN.fit(x_train, y_train)
    outKN = clrKN.predict(x_test)

    # Prediction accuracy
    print("Accuracy for Decision Tree Classifier: " + str(accuracy_score(y_test, outTree)*100)+"%")
    print("Accuracy for KNeighbors Classifier: " + str(accuracy_score(y_test, outKN)*100)+"%")

main()

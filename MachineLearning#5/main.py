from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


# Finds Euclidean Distance
def euc(a, b):
    return distance.euclidean(a, b)


# New Classifier
class NewClassifier:

    x_train = []
    y_train = []

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        prediction = []

        for row in x_test:
            label = self.closest(row)
            prediction.append(label)

        return prediction

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


# Main Method
def main():

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)
    clr = NewClassifier()
    clr.fit(x_train, y_train)
    prediction = clr.predict(x_test)

    # Prediction accuracy
    print("Accuracy: " + str(accuracy_score(y_test, prediction) * 100) + "%")


# Run main
main()
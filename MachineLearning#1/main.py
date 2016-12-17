from sklearn import tree


def main():
    #  0: smooth, 1: bumpy
    features = [[130, 0], [140, 0], [150, 1], [170, 1]]

    # 0: apple, 1: orange
    labels = [0, 0, 1, 1]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)

    # 160, smooth
    predict = [[160, 0]]

    if (clf.predict(predict)[0]) == int(0):
        print('you are describing orange')
    elif (clf.predict(predict)[0]) == int(1):
        print('you are describing apple')
    else:
        print('Can\'t Guess')

main()

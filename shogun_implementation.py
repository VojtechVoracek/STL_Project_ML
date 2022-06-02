from shogun import *
from sklearn.metrics import confusion_matrix

def linear_regression(x_train, x_test, y_train, y_test):

    x_train = RealFeatures(x_train.T)
    x_test = RealFeatures(x_test.T)
    y_train = RegressionLabels(y_train.T[0])
    y_test = RegressionLabels(y_test.T[0])

    clf = LeastSquaresRegression(x_train, y_train)
    clf.train()
    labels_predict = clf.apply_regression(x_test)
    mse = MeanSquaredError().evaluate(labels_predict, y_test)

    return mse


def k_nn(x_train, x_test, y_train, y_test, k):

    x_train = RealFeatures(x_train.T)
    x_test = RealFeatures(x_test.T)
    y_train = MulticlassLabels(y_train.T[0])
    y_test = MulticlassLabels(y_test.T[0])

    distance = EuclideanDistance(x_train, x_train)

    knn = KNN(k, distance, y_train)
    knn.train()
    labels_predict = knn.apply_multiclass(x_test)

    acc = MulticlassAccuracy().evaluate(labels_predict, y_test)
    conf_mat = MulticlassAccuracy.get_confusion_matrix(labels_predict, y_test)

    return acc, conf_mat
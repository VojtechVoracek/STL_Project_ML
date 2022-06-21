from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def linear_regression(x_train, x_test, y_train, y_test):

    clf = LinearRegression()
    clf.fit(x_train, y_train)

    mse = metrics.mean_squared_error(x_test @ clf.coef_.T + clf.intercept_, y_test)

    return mse


def k_nn(x_train, x_test, y_train, y_test, k):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train.T[0])
    pred = knn.predict(x_test)

    acc = metrics.accuracy_score(y_test, pred)
    conf_mat = metrics.confusion_matrix(y_test, pred)

    return acc, conf_mat

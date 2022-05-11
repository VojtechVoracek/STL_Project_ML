import numpy as np
from LoadDatasets import *
import shogun_implementation
import sklearn_implementation


def median_mse(dataset_getter_function, lin_reg_func, num_of_runs):

    mses = np.empty(num_of_runs)

    for i in range(num_of_runs):
        x_train, x_test, y_train, y_test = dataset_getter_function()
        mses[i] = lin_reg_func(x_train, x_test, y_train, y_test)

    return np.median(mses)


def median_acc(dataset_getter_function, k_nn_func, num_of_runs, k):

    accs = np.empty(num_of_runs)

    for i in range(num_of_runs):
        x_train, x_test, y_train, y_test = dataset_getter_function()
        accs[i] = k_nn_func(x_train, x_test, y_train, y_test, k)

    return np.median(accs)


if __name__ == '__main__':

    num_of_runs = 25
    k_iris = 7
    k_yeast = 20

    print("Shotgun:")
    diabetes_mse = median_mse(get_diabetes, shogun_implementation.linear_regression, num_of_runs)
    diamonds_mse = median_mse(get_diamonds, shogun_implementation.linear_regression, num_of_runs)
    iris_acc = median_acc(get_iris, shogun_implementation.k_nn, num_of_runs, k_iris)
    yeast_acc = median_acc(get_yeast, shogun_implementation.k_nn, num_of_runs, k_yeast)
    print("Diabetes dataset: median test set MSE over 25 runs:", diabetes_mse)
    print("Diamonds dataset: median test set MSE over 25 runs:", diamonds_mse)
    print("Iris dataset: median test set ACC over 25 runs:", iris_acc)
    print("Yeast dataset: median test set ACC over 25 runs:", yeast_acc)
    print("--------------------------------------------------------------")

    print("Scikit-learn:")
    diabetes_mse2 = median_mse(get_diabetes, sklearn_implementation.linear_regression, num_of_runs)
    diamonds_mse2 = median_mse(get_diamonds, sklearn_implementation.linear_regression, num_of_runs)
    iris_acc2 = median_acc(get_iris, sklearn_implementation.k_nn, num_of_runs, k_iris)
    yeast_acc2 = median_acc(get_yeast, sklearn_implementation.k_nn, num_of_runs, k_yeast)
    print("Diabetes dataset: median test set MSE over 25 runs:", diabetes_mse2)
    print("Diamonds dataset: median test set MSE over 25 runs:", diamonds_mse2)
    print("Iris dataset: median test set ACC over 25 runs:", iris_acc2)
    print("Yeast dataset: median test set ACC over 25 runs:", yeast_acc2)
    print("--------------------------------------------------------------")

    print("Dataset         shogun             sklearn")
    print("Diabetes |", diabetes_mse, "|", diabetes_mse2)
    print("Diamond  |", diamonds_mse, "|", diamonds_mse2)
    print("Iris     |", iris_acc, "|", iris_acc2)
    print("Yeast    |", yeast_acc, "|", yeast_acc2)
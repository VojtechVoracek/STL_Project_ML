import numpy as np
import pandas as pd

from LoadDatasets import *
import shogun_implementation
import sklearn_implementation
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm

def median_mse(dataset_getter_function, lin_reg_func, num_of_runs):

    mses = np.empty(num_of_runs)

    for i in range(num_of_runs):
        x_train, x_test, y_train, y_test = dataset_getter_function()
        mses[i] = lin_reg_func(x_train, x_test, y_train, y_test)

    return np.median(mses)


def median_acc(dataset_getter_function, k_nn_func, num_of_runs, k, num_of_classes):

    accs = np.empty(num_of_runs)
    conf_mat = np.zeros((num_of_classes, num_of_classes, num_of_runs))

    for i in range(num_of_runs):
        x_train, x_test, y_train, y_test = dataset_getter_function()
        accs[i], conf_mat[:, :, i] = k_nn_func(x_train, x_test, y_train, y_test, k)

    return np.median(accs), np.median(conf_mat, axis=2)


def plot_cm(cm, title):

    if cm.shape[0] == 10:
        df_cm = pd.DataFrame(cm, index=[i for i in ["CYT", "ERL", "EXC", "ME1", "ME2", "ME3", "MIT", "NUC", "POX", "VAC"]],
                         columns=[i for i in ["CYT", "ERL", "EXC", "ME1", "ME2", "ME3", "MIT", "NUC", "POX", "VAC"]])
    else:
        df_cm = pd.DataFrame(cm, index=[i for i in ["setosa", "versicolor", "virginica"]],
                             columns=[i for i in ["setosa", "versicolor", "virginica"]])

    plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=30, y=1.01)
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu", norm=LogNorm(), cbar=False)

    plt.savefig("figures/" + title + ".png")



if __name__ == '__main__':
    sn.set(font_scale=1.4)
    num_of_runs = 25
    k_iris = 7
    k_yeast = 20

    print("Shogun:")
    diabetes_mse = median_mse(get_diabetes, shogun_implementation.linear_regression, num_of_runs)
    diamonds_mse = median_mse(get_diamonds, shogun_implementation.linear_regression, num_of_runs)
    iris_acc, iris_cm = median_acc(get_iris, shogun_implementation.k_nn, num_of_runs, k_iris, 3)
    yeast_acc, yeast_cm = median_acc(get_yeast, shogun_implementation.k_nn, num_of_runs, k_yeast, 10)
    print("Diabetes dataset: median test set MSE over 25 runs:", diabetes_mse)
    print("Diamonds dataset: median test set MSE over 25 runs:", diamonds_mse)
    print("Iris dataset: median test set ACC over 25 runs:", iris_acc)
    print("Yeast dataset: median test set ACC over 25 runs:", yeast_acc)
    print("--------------------------------------------------------------")

    print("Scikit-learn:")
    diabetes_mse2 = median_mse(get_diabetes, sklearn_implementation.linear_regression, num_of_runs)
    diamonds_mse2 = median_mse(get_diamonds, sklearn_implementation.linear_regression, num_of_runs)
    iris_acc2, iris_cm2 = median_acc(get_iris, sklearn_implementation.k_nn, num_of_runs, k_iris, 3)
    yeast_acc2, yeast_cm2 = median_acc(get_yeast, sklearn_implementation.k_nn, num_of_runs, k_yeast, 10)
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

    plot_cm(iris_cm, "Iris - shogun")
    plot_cm(iris_cm2, "Iris - scikit-learn")
    plot_cm(yeast_cm, "Yeast - shogun")
    plot_cm(yeast_cm2, "Yeast - scikit-learn")

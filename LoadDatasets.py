import pandas as pd
from sklearn.model_selection import train_test_split


def get_diabetes():
    x = pd.read_csv("prepared_datasets/diabetes.data.csv").to_numpy()
    y = pd.read_csv("prepared_datasets/diabetes.labels.csv").to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    return x_train, x_test, y_train, y_test


def get_diamonds():
    x = pd.read_csv("prepared_datasets/diamonds.data.csv").to_numpy()
    y = pd.read_csv("prepared_datasets/diamonds.labels.csv").to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    return x_train, x_test, y_train, y_test


def get_iris():
    x = pd.read_csv("prepared_datasets/iris.data.csv").to_numpy()
    y = pd.read_csv("prepared_datasets/iris.labels.csv").to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    return x_train, x_test, y_train, y_test


def get_yeast():
    x = pd.read_csv("prepared_datasets/yeast.data.csv").to_numpy()
    y = pd.read_csv("prepared_datasets/yeast.labels.csv").to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    return x_train, x_test, y_train, y_test
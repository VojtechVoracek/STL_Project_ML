import numpy as np
import pandas as pd


def prepare_iris(f_name, name):
    data = pd.read_csv(f_name).to_numpy()

    x = data[:, 1:-1]
    y = data[:, -1]

    class_names = np.unique(y)
    for i in range(len(class_names)):
        class_name = class_names[i]
        y[np.where(y == class_name)[0]] = i

    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs)
    x = x[idxs]
    y = y[idxs]

    f_data = "prepared_datasets/" + name + ".data.csv"
    f_labels = "prepared_datasets/" + name + ".labels.csv"
    np.savetxt(f_data, x, delimiter=",")
    np.savetxt(f_labels, y, delimiter=",")


def prepare_yeast(f_name, name):
    data = pd.read_csv(f_name).to_numpy()

    x = data[:, 0:-1]
    y = data[:, -1]
    class_names = np.unique(y)
    for i in range(len(class_names)):
        class_name = class_names[i]
        y[np.where(y == class_name)[0]] = i

    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs)
    x = x[idxs]
    y = y[idxs]

    f_data = "prepared_datasets/" + name + ".data.csv"
    f_labels = "prepared_datasets/" + name + ".labels.csv"
    np.savetxt(f_data, x, delimiter=",")
    np.savetxt(f_labels, y, delimiter=",")


def prepare_diabetes(f_name, name):
    data = pd.read_csv(f_name, sep='\t').to_numpy()

    x = data[:, 0:-1]
    y = data[:, -1]
    class_names = np.unique(y)
    for i in range(len(class_names)):
        class_name = class_names[i]
        y[np.where(y == class_name)[0]] = i

    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs)
    x = x[idxs]
    y = y[idxs]

    f_data = "prepared_datasets/" + name + ".data.csv"
    f_labels = "prepared_datasets/" + name + ".labels.csv"
    np.savetxt(f_data, x, delimiter=",")
    np.savetxt(f_labels, y, delimiter=",")


def prepare_diamonds(f_name, name):
    data = pd.read_csv(f_name).to_numpy()

    x = data[:, np.array([1,2,3,4,5,6,8,9,10])]
    y = data[:, 7]

    quality = np.array(["Fair", "Good", "Very Good", "Premium", "Ideal"])

    for i in range(len(quality)):
        idxs = np.where(x[:, 1] == quality[i])
        x[idxs, 1] = i

    color = np.array(["J", "I", "H", "G", "F", "E", "D"])

    for i in range(len(color)):
        idxs = np.where(x[:, 2] == color[i])
        x[idxs, 2] = i

    clarity = np.array(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

    for i in range(len(clarity)):
        idxs = np.where(x[:, 3] == clarity[i])
        x[idxs, 3] = i

    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs)
    x = x[idxs]
    y = y[idxs]

    f_data = "prepared_datasets/" + name + ".data.csv"
    f_labels = "prepared_datasets/" + name + ".labels.csv"
    np.savetxt(f_data, x, delimiter=",")
    np.savetxt(f_labels, y, delimiter=",")


if __name__ == "__main__":

    f_name_iris = "datasets/k-nn/toy-Iris.csv"
    f_name_yeast = "datasets/k-nn/yeast.csv"
    f_name_diabetes = "datasets/linear_regression/toy-diabetes.csv"
    f_name_diamonds = "datasets/linear_regression/diamonds.csv"

    prepare_iris(f_name_iris, "iris")
    prepare_yeast(f_name_yeast, "yeast")
    prepare_diamonds(f_name_diamonds, "diamonds")
    prepare_diabetes(f_name_diabetes, "diabetes")

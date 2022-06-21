This program performs the knn classification task on a given dataset in three steps:

1) the dataset is split into training and test set.

2) the knn algorithm is applied to the two datasets. The resulting neighbours for each test sample and the corresponding distances are stored in the folder "neighbours_data".

3) the assignment of the test samples is done on an own c++ implementation (via majority vote). The resulting labels of the training data are stored in the folder "predicted_data".


To run the program execute:

bash run_knn

There are several flags for choosing the dataset, choosing the test set size, choosing the parameter k and switching on/off output to the console:

-d <dataset> : this option can contain the string 'iris' or 'yeast' for <dataset> and defines the used dataset (default=iris)
-t <testsize> : this option can contain a value for setting the size of the test set, I.e. the fraction of the original dataset which is used for testing (default = 0.2)
-k : this option can contain an integer which defines the amount of nearest neighbours which are taken into account for each test sample. (default = 5)
-v : if this option is set, the informational messages and the full list of parameters and timers of the mlpack function are displayed in the console.

For Example, to run the program on the yeast dataset with test set size of 0.3 and k=20 you can run:

bash run_knn -d yeast -t 0.3 -k 20




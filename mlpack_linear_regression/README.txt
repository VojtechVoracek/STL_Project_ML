This program performs the linear regression task on a given dataset in three steps:

1) the dataset is split into training and test set. The test set will be used to predict the dependent variable in order to be able to calculate the accuracy metric.

2) the linear regression algorithm is applied to the two datasets. The resulting values of the dependent variable for each test sample are stored in the folder "predicted_data".

3) The mean squared error of the predicted values is computed and print out to the terminal.


To run the program execute:

bash run_linear_regression

There are several flags for choosing the dataset, choosing the test set size and switching on/off output to the console:

-d <dataset> : this option can contain the string 'diabetes' or 'diamonds' for <dataset> and defines the used dataset (default = diabetes)
-t <testsize> : this option can contain a value for setting the size of the test set, i.e. the fraction of the original dataset which is used for testing (default = 0.2)
-v : if this option is set, the informational messages and the full list of parameters and timers of the mlpack function are displayed in the console.

For Example, to run the program on the diamonds dataset with test set size of 0.3 you can run:

bash run_linear_regression -d diamonds -t 0.3




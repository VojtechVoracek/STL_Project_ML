#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <numeric>
#include <mlpack/core.hpp>


void readDataFromFile(std::string filepath, std::vector<std::vector<double>>& features){
    // File pointer
    std::fstream fin;
    // Open an existing file
    fin.open(filepath, std::fstream::in);

    std::vector<double> row;
    std::string line, element;
    // create stingstream
    std::istringstream ss_row;
    while(std::getline(fin, line)){

            // remove all entries from row
			row.clear();
            // prepare ss_row for reuse
            ss_row.clear();
            // assign line to stringstream
            ss_row.str(line);
    
			while(std::getline(ss_row, element, ',')){
	
                row.push_back(std::stod(element));
            }
			features.push_back(row);
		}
    fin.close();
}

double compute_MSE(std::vector<double>& true_labels, std::vector<double>& predicted_labels){
    double mse = 0;
    int no_rows = true_labels.size();
    
    for(int i = 0; i < no_rows; i++){
        mse = mse + pow(true_labels[i] - predicted_labels[i],2);
    }
    mse = mse / no_rows;
    return mse;
}

double compute_R_squared(std::vector<double>& true_labels, std::vector<double>& predicted_labels){
    double r_squared = 0, rss = 0, tss=0;
    int no_rows = true_labels.size();
    double true_average = std::accumulate( true_labels.begin(), true_labels.end(), 0.0) / no_rows;

    for(int i = 0; i < no_rows; i++){
        rss = rss + pow(true_labels[i] - predicted_labels[i],2);
        tss = tss + pow(true_labels[i] - true_average,2);
    }
    r_squared = 1 - (rss / tss);
    return r_squared;
}


int main(int argc, char** argv){
    // argc (argument count), argv (argument vector) 
    // the first element of argv is the name of the program
    std::string dataset = "diabetes";
    if(argc > 1){
        dataset = argv[1];
    }

    std::vector<std::vector<double>> test_labels;
    std::string path_test_labels = "../preprocessed_data/" +dataset+ ".labels.test.csv";

    std::vector<std::vector<double>> test_labels_predicted;
    std::string path_test_labels_predicted = "../predicted_data/" +dataset+ ".labels.test.predicted.csv";

    readDataFromFile(path_test_labels, test_labels);
    readDataFromFile(path_test_labels_predicted, test_labels_predicted);

   
    int length = test_labels.size();
    // prepare vector of vectors because it needs to be stored as one column vector
    std::vector<double> test_labels_prepared;
    std::vector<double> test_labels_predicted_prepared;
    for(int i = 0; i<length; i++){
        test_labels_prepared.push_back(test_labels[i][0]);
        test_labels_predicted_prepared.push_back(test_labels_predicted[i][0]);
    }

    // this code part transforms a vector into an arma::mat object.
    /* 
    arma::mat data(test_labels_propared);
    data.print();
    std::cout << data.n_rows << ' ' << data.n_cols << ' ' << data.n_elem; */
    double r_squared = compute_R_squared(test_labels_prepared,test_labels_predicted_prepared);
    double mse = compute_MSE(test_labels_prepared,test_labels_predicted_prepared);
    std::cout << "Evaluation of test prediction on " +dataset+ " dataset:" << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;
    std::cout << "Root Mean Squared Error: " << std::sqrt(mse) << std::endl;
    std::cout << "R^2: " << r_squared << std::endl;

    return 0;
}
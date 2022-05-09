#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
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


int main(){

    std::vector<std::vector<double>> diabetes_test_labels;
    std::string path_diabetes_test_labels = "../preprocessed_data/diabetes.labels.test.csv";

    std::vector<std::vector<double>> diabetes_test_labels_predicted;
    std::string path_diabetes_test_labels_predicted = "../predicted_data/diabetes.labels.test.predicted.csv";

    readDataFromFile(path_diabetes_test_labels, diabetes_test_labels);
    readDataFromFile(path_diabetes_test_labels_predicted, diabetes_test_labels_predicted);

    double mse = compute_MSE(diabetes_test_labels[0],diabetes_test_labels_predicted[0]);
    std::cout << mse;

    return 0;
}
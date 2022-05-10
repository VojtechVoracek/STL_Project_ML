#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
//#include <algorithm>
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

int mostFrequent(int arr[], int n){
    // sort the array ascendingly
    std::sort(arr, arr + n);

    // Find the max frequency using linear traversal
    int max_count = 1, res = arr[0], curr_count = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i] == arr[i - 1])
            curr_count++;
        else
            curr_count = 1;
    
        if (curr_count > max_count) {
            max_count = curr_count;
            res = arr[i - 1];
        }
    }

    return res;

}

void predictClasses(std::vector<std::vector<double>>& neighbour_ids, std::vector<std::vector<double>>& train_labels, int no_test_samples, int k, int predicted_classes[]){
    // allocate memory
    int* labels = new int[k];

    for(int i = 0; i<no_test_samples; i++){
        for(int j = 0; j<k; j++){
            labels[j] = train_labels[neighbour_ids[i][j]][0];
        }
        predicted_classes[i] = mostFrequent(labels, k);
    }
    // deallocate memory
    delete[] labels;
}

void writeDataToFile(std::string filepath, int data[], int data_length){
    // data must be one dimensional
    std::ofstream output_file;
    output_file.open(filepath);
    for(int i = 0; i<data_length; i++){
        output_file << std::to_string(data[i]) << '\n';
    }

    output_file.close();
}

double claculateAccuracy(int predicted_labels[], int true_labels[], int testsize){
    double accuracy = 0;
    for(int i = 0; i<testsize; i++){
        if(predicted_labels[i] == true_labels[i]){
            accuracy++;
        }
    }
    accuracy = accuracy / testsize;
    return accuracy;
}

int main(int argc, char** argv){
    std::string dataset = "iris";
    if(argc > 1){
        dataset = argv[1];
    }

    std::vector<std::vector<double>> neighbour_ids;
    std::string path_neighbour_ids = "../neighbours_data/" +dataset+ ".neighbours.test.csv";
    std::vector<std::vector<double>> train_labels;
    std::string path_train_labels = "../preprocessed_data/" +dataset+ ".labels.train.csv";
    std::vector<std::vector<double>> test_labels;
    std::string path_test_labels = "../preprocessed_data/" +dataset+ ".labels.test.csv";
    
    // read the neighbour_ids and the training labels from the csv files, created by the mlpack functions in the bash script
    readDataFromFile(path_neighbour_ids, neighbour_ids);
    readDataFromFile(path_train_labels, train_labels);
    readDataFromFile(path_test_labels, test_labels);
    
    // get the size of the data
    int no_test_samples = neighbour_ids.size();
    int k = neighbour_ids[0].size();
    
    // predict the labels and save it in an integer array
    int* predicted_classes = new int[no_test_samples];
    predictClasses(neighbour_ids, train_labels, no_test_samples, k, predicted_classes);

    // store the predicted labels in a file
    std::string filepath = "../predicted_data/"+dataset+".labels.test.predicted.csv";
    writeDataToFile(filepath, predicted_classes, no_test_samples);
    
    // calculate the accuracy as the fraction of correctly classified samples
    int* true_labels = new int[no_test_samples];
    for(int i = 0; i<no_test_samples; i++){
        true_labels[i] = int(test_labels[i][0]);
    }
    double accuracy = claculateAccuracy(predicted_classes, true_labels, no_test_samples);
    std::cout << "Accuracy on the test set: " + std::to_string(accuracy) << std::endl;


    // delete allocated memory
    delete[] predicted_classes;


    return 0;
}
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

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

int main(int argc, char** argv){
    std::string dataset = "iris";
    if(argc > 1){
        dataset = argv[1];
    }

    //std::vector<std::vector<double>> neighbour_distances;
    //std::string path_neighbour_distances = "../neighbours_data/" +dataset+ ".distances.test.csv";

    std::vector<std::vector<double>> neighbour_ids;
    std::string path_neighbour_ids = "../neighbours_data/" +dataset+ ".neighbours.test.csv";
    
    
    std::vector<std::vector<double>> train_labels;
    std::string path_train_labels = "../preprocessed_data/" +dataset+ ".labels.train.csv";
    

    readDataFromFile(path_neighbour_ids, neighbour_ids);
    readDataFromFile(path_train_labels, train_labels);
    
    int no_test_samples = neighbour_ids.size();
    int predicted_classes[no_test_samples] = {};


    std::cout << no_test_samples << std::endl;

    return 0;
}
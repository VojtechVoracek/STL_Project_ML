#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>


#include <mlpack/core.hpp>


void readDataFromFile(std::string filepath, std::vector<std::vector<double>>& features){
// File pointer
    std::fstream fin;
    // create filepath to toy-dataset

    // Open an existing file
    fin.open(filepath, std::fstream::in);

    std::vector<double> row;
    std::string line, element;
    // create stingstream
    std::istringstream ss_row;
    while(std::getline(fin, line))
		{
            // remove all entries from row
			row.clear();
            // prepare ss_row for reuse
            ss_row.clear();
            // assign line to stringstream
            ss_row.str(line);

			while(std::getline(ss_row, element, ',')){
                // parse element to double and add element to row
                row.push_back(std::stod(element));
            }
            // add row to features
			features.push_back(row);
		}
    // close the file
    fin.close();
}


int main(){
    std::vector<std::vector<double>> diabetes_features;
    std::string filepath_toy = "../../prepared_datasets/diabetes.data.csv";

    readDataFromFile(filepath_toy, diabetes_features);




    int no_rows = diabetes_features.size();
    int no_columns = diabetes_features[0].size();
    //std::cout << diabetes_features[0][0].size();
    for(int i = 0; i < no_rows; i++){
        for(int j = 0; j < no_columns; j++){
            std::cout << diabetes_features[i][j] << ' ';
        }
        std::cout  << std::endl;
    }
    return 0;
}
#include <string>
#include <fstream>
#include <sstream>
#include "mnist_iterator.h"

using std::stringstream;
using std::ifstream;
using std::getline;

const string mnist_iterator::PATH_TRAIN = "data/mnist_train.csv";
const string mnist_iterator::PATH_TEST = "data/mnist_test.csv";
const int mnist_iterator::TYPE_TRAIN = 1;
const int mnist_iterator::TYPE_TEST = 2;

double **mnist_iterator::get_inputs(int index, int batch_size, int sample_type) const{
	ifstream file;
	file.open(sample_type == TYPE_TRAIN ? PATH_TRAIN : PATH_TEST);
	string line;
	for(int i = 0; i < index; i++){
		getline(file, line);
	}
	
	double **batch = new double*[batch_size];
	for(int i = 0; i < batch_size; i++){
		getline(file, line);
		stringstream ss(line);
		int val;
		ss >> val;//skip the label
		batch[i] = new double[28*28];
		for(int j = 0; j < 28*28; j++){
			ss.ignore();
			ss >> val;
			batch[i][j] = val / 255.0;
		}
	}

	file.close();
	return batch;
}

double **mnist_iterator::get_desired_outputs(int index, int batch_size, int sample_type) const{
	ifstream file;
	file.open(sample_type == TYPE_TRAIN ? PATH_TRAIN : PATH_TEST);
	string line;
	for(int i = 0; i < index; i++){
		getline(file, line);
	}

	double **desired_outputs = new double*[batch_size];
	for(int i = 0; i < batch_size; i++){
		getline(file, line);
		stringstream ss(line);
		int val;
		ss >> val;
		desired_outputs[i] = new double[10];
		for(int j = 0; j < 10; j++){
			if(j == val)
				desired_outputs[i][j] = 1.0;
			else
				desired_outputs[i][j] = 0.0;
		}
	}

	file.close();
	return desired_outputs;
}

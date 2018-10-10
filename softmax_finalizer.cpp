#include "softmax_finalizer.h"

void softmax_finalizer::operator()(double *output_activations, int num_activations) const{
	double sum = 0;
	for(int i = 0; i < num_activations; i++){
		sum += output_activations[i];
	}
	for(int i = 0; i < num_activations; i++){
		output_activations[i] /= sum;
	}
}

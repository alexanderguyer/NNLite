#include <math.h>
#include <cstddef>
#include "sigmoid_derivative.h"

sigmoid_derivative::sigmoid_derivative(){
	derivative = NULL;
	m_finalizer = NULL;
}

activation_function *sigmoid_derivative::clone() const{
	return new sigmoid_derivative;
}

double sigmoid_derivative::operator()(double x) const{
	return exp(-x) / pow(1 + exp(-x), 2);
}

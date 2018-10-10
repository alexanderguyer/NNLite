#include <cstddef>
#include <math.h>
#include "sigmoid.h"
#include "sigmoid_derivative.h"

sigmoid::sigmoid(){
	derivative = new sigmoid_derivative;
	m_finalizer = NULL;
}

activation_function *sigmoid::clone() const{
	return new sigmoid;
}

double sigmoid::operator()(double x) const{
	return 1.0 / (1.0 + exp(-x));
}

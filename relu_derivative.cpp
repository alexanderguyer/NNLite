#include <cstddef>
#include "relu_derivative.h"

relu_derivative::relu_derivative(){
	derivative = NULL;
	m_finalizer = NULL;
}

activation_function *relu_derivative::clone() const{
	return new relu_derivative;
}

double relu_derivative::operator()(double input) const{
	return input >= 0 ? 1 : 0;
}

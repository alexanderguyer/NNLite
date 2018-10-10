#include <cstddef>
#include "relu.h"
#include "relu_derivative.h"

relu::relu(){
	derivative = new relu_derivative;
	m_finalizer = NULL;
}

activation_function *relu::clone() const{
	return new relu;
};

double relu::operator()(double input) const{
	return input > 0 ? input : 0;
}

#include "mse.h"

double mse::operator()(double input, double activation, double desired_activation) const{
	return input * (activation - desired_activation);
}

loss_function *mse::clone() const{
	return new mse;
}

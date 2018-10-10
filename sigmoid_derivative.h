#ifndef SIGMOID_DERIVATIVE
#define SIGMOID_DERIVATIVE

#include "activation_function.h"

class sigmoid_derivative : public activation_function{
public:
	sigmoid_derivative();

	virtual activation_function *clone() const;
	virtual double operator()(double) const;
};

#endif

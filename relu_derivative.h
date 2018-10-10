/**

	RELU DERIVATIVE ACTIVATION FUNCTION
	Created by Alexander Guyer on 9/4/2018 at 10:34 AM

*/

#ifndef RELU_DERIVATIVE_H
#define RELU_DERIVATIVE_H

#include "activation_function.h"

class relu_derivative : public activation_function{
public:
	relu_derivative();
	virtual activation_function *clone() const;
	virtual double operator()(double) const;
};

#endif

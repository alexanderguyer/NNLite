/**

	RELU ACTIVATION FUNCTION
	Created by Alexander Guyer on 9/4/2018 at 10:29 AM

*/

#ifndef RELU_H
#define RELU_H

#include "activation_function.h"

class relu : public activation_function{
public:
	relu();
	virtual activation_function *clone() const;
	virtual double operator()(double) const;
};

#endif

#ifndef SIGMOID_H
#define SIGMOID_H

#include "activation_function.h"

class sigmoid : public activation_function{
public:
	sigmoid();
	virtual activation_function *clone() const;
	virtual double operator()(double) const;
};

#endif

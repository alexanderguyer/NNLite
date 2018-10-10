#ifndef ACTIVATION_FUNCTION
#define ACTIVATION_FUNCTION

#include "finalizer.h"

class activation_function{
protected:
	activation_function *derivative;
	finalizer *m_finalizer;
public:
	~activation_function();

	activation_function *get_derivative() const;
	finalizer *get_finalizer() const;

	virtual activation_function *clone() const = 0;
	virtual double operator()(double) const = 0;
};

#endif

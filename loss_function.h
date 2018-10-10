#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

class loss_function{
	public:
		virtual double operator()(double, double, double) const = 0;
		virtual loss_function *clone() const = 0;
};

#endif

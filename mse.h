/**

	MEAN SUM OF ERRORS
	Created by Alexander Guyer

*/

#ifndef MSE_H
#define MSE_H

#include "loss_function.h"

class mse : public loss_function{
	public:
		virtual double operator()(double, double, double) const;
		virtual loss_function *clone() const;
};

#endif

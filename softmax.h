/**

	SOFTMAX ACTIVATION FUNCTION
	Created by Alexander Guyer on 9/2/2018 at 8:52 AM

*/

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "sigmoid.h"

class softmax : public sigmoid{
public:
	softmax();
	virtual activation_function *clone() const;
};

#endif

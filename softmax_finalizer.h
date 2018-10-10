/**

	SOFTMAX FINALIZER
	Created by Alexander Guyer on 9/2/2018 at 8:58 AM

*/

#ifndef SOFTMAX_FINALIZER_H
#define SOFTMAX_FINALIZER_H

#include "finalizer.h"

class softmax_finalizer : public finalizer{
public:
	virtual void operator()(double *, int) const;
};

#endif

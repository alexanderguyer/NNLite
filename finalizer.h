#ifndef FINALIZER_H
#define FINALIZER_H

class finalizer{
public:
	virtual void operator()(double *, int) const = 0;
};

#endif

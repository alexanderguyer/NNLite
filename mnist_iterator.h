#ifndef MNIST_ITERATOR_H
#define MNIST_ITERATOR_H

#include <string>
#include "sample_iterator.h"

using std::string;

class mnist_iterator : public sample_iterator{
	public:
		static const string PATH_TRAIN;
		static const string PATH_TEST;
		static const int TYPE_TRAIN;
		static const int TYPE_TEST;
		virtual double **get_inputs(int, int, int) const;
		virtual double **get_desired_outputs(int, int, int) const;
};

#endif

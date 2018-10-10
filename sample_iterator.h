#ifndef SAMPLE_ITERATOR_H
#define SAMPLE_ITERATOR_H

class sample_iterator{
	public:
		virtual double **get_inputs(int, int, int = 0) const = 0;
		virtual double **get_desired_outputs(int, int, int = 0) const = 0;
};

#endif

#include "softmax.h"
#include "softmax_finalizer.h"

softmax::softmax() : sigmoid(){
	m_finalizer = new softmax_finalizer;
}

activation_function *softmax::clone() const{
	return new softmax;
}

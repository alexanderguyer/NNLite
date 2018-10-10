#include "activation_function.h"

activation_function::~activation_function(){
	delete derivative;
	delete m_finalizer;
}

activation_function *activation_function::get_derivative() const{
	return derivative;
}

finalizer *activation_function::get_finalizer() const{
	return m_finalizer;
}

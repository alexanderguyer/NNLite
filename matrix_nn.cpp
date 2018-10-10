#include <cstdlib>
#include <iostream>
#include "matrix_nn.h"

using std::cout;

//TODO: Add learning rate decay with backprop.
//TODO: Modify softmax finalizer so that it makes sure all values are non-negative by subtracting the smallest value from all activations.
//TODO: Optimize code, add more constructors / options, make dropout something that can be enabled / disabled, make functions to modify the structure of the neural network, such as add_layer, add node, etc.

matrix_nn::matrix_nn(int num_layers, int *num_nodes, activation_function **activation_functions, loss_function *m_loss_function, double training_rate, double *p_keep){
	this->p_keep = NULL;
	rprop_updates = NULL;
	rprop_signs = NULL;
	first_rprop = true;
	inputs = NULL;
	desired_activations = NULL;
	this->num_layers = num_layers;
	this->num_nodes = new int[num_layers];
	for(int i = 0; i < num_layers; i++){
		this->num_nodes[i] = num_nodes[i];
	}

	weights = new double**[num_layers - 1];
	for(int i = 0; i < num_layers - 1; i++){
		weights[i] = new double*[num_nodes[i]];
		for(int j = 0; j < num_nodes[i]; j++){
			weights[i][j] = new double[num_nodes[i + 1]];
			double iter = 0;
			double delta = 0.01 / num_nodes[i + 1];
			for(int k = 0; k < num_nodes[i + 1]; k++){
				weights[i][j][k] = (rand() % 1001) / 100000.0;
				//weights[i][j][k] = iter + ((rand() % 2001 - 1000) / 1000.0) * 0.001 * delta;
				iter += delta;
			}
		}
	}
	
	node_inputs = new double**[num_layers - 1];
	node_activations = new double**[num_layers - 1];
	for(int i = 0; i < num_layers - 1; i++){
		node_inputs[i] = NULL;
		node_activations[i] = NULL;
	}

	this->activation_functions = new activation_function*[num_layers - 1];
	for(int i = 0; i < num_layers - 1; i++){
		this->activation_functions[i] = activation_functions[i]->clone();
	}

	num_sets = 0;

	this->m_loss_function = m_loss_function->clone();
	this->training_rate = training_rate;

	input_gradients = new double**[num_layers - 1];
	activation_gradients = new double**[num_layers - 2];//We don't care about the activation gradients for the first layer, since they are never used.
	for(int i = 0; i < num_layers - 2; i++){
		input_gradients[i] = NULL;
		activation_gradients[i] = NULL;
	}
	input_gradients[num_layers - 2] = NULL;

	update_p_keep(p_keep);
	keep_mask = new bool**[num_layers - 2];
	keep_mask[0] = NULL;
}


matrix_nn::~matrix_nn(){
	delete_rprop_updates();
	delete_backward_products();
	delete [] input_gradients;
	delete [] activation_gradients;
	delete m_loss_function;

	for(int i = 0; i < num_layers - 1; i++){
		delete activation_functions[i];
	}
	delete [] activation_functions;
	
	delete_inputs();
	delete [] this->inputs;
	delete_desired_activations();
	delete [] this->desired_activations;

	delete_forward_products();
	delete [] node_inputs;
	delete [] node_activations;

	for(int i = 0; i < num_layers - 1; i++){
		for(int j = 0; j < num_nodes[i]; j++){
			delete [] weights[i][j];
		}
		delete [] weights[i];
	}
	delete [] weights;
	delete [] num_nodes;
	delete [] p_keep;
	delete_keep_mask();
	delete [] keep_mask;
}

void matrix_nn::update_inputs(double **inputs){
	if(num_sets > 0){
		this->inputs = new double*[num_sets];
		for(int i = 0; i < num_sets; i++){
			this->inputs[i] = new double[num_nodes[0]];
			for(int j = 0; j < num_nodes[0]; j++){
				this->inputs[i][j] = inputs[i][j];
			}
		}
	}
}

void matrix_nn::delete_inputs(){
	if(this->inputs != NULL){
		for(int i = 0; i < this->num_sets; i++){
			delete [] this->inputs[i];
		}
		delete [] this->inputs;
		this->inputs = NULL;
	}
}

void matrix_nn::update_desired_activations(double **desired_activations){
	if(num_sets > 0){
		this->desired_activations = new double*[num_sets];
		for(int i = 0; i < num_sets; i++){
			this->desired_activations[i] = new double[num_nodes[num_layers - 1]];
			for(int j = 0; j < num_nodes[num_layers - 1]; j++){
				this->desired_activations[i][j] = desired_activations[i][j];
			}
		}
	}
}

void matrix_nn::delete_desired_activations(){
	if(this->desired_activations != NULL){
		for(int i = 0; i < this->num_sets; i++){
			delete [] this->desired_activations[i];
		}
		delete [] this->desired_activations;
		this->desired_activations = NULL;
	}
}

void matrix_nn::update_num_sets(int num_sets){
	delete_inputs();
	delete_desired_activations();
	delete_keep_mask();
	delete_forward_products();
	delete_backward_products();
	this->num_sets = num_sets;
}

void matrix_nn::update_sets(double **inputs, double **desired_activations, int num_sets){
	update_num_sets(num_sets);
	update_inputs(inputs);
	update_desired_activations(desired_activations);
	gen_keep_mask();
}

void matrix_nn::delete_rprop_updates(){
	if(rprop_updates != NULL){
		for(int i = 0; i < num_layers - 1; i++){
			for(int j = 0; j < num_nodes[i]; j++){
				delete [] rprop_updates[i][j];
				delete [] rprop_signs[i][j];
			}
			delete [] rprop_updates[i];
			delete [] rprop_signs[i];
		}
		delete [] rprop_updates;
		delete [] rprop_signs;
	}
}

void matrix_nn::update_rprop_updates(double ***rprop_updates){
	delete_rprop_updates();
	this->rprop_updates = new double**[num_layers - 1];
	rprop_signs = new char**[num_layers - 1];
	for(int i = 0; i < num_layers - 1; i++){
		this->rprop_updates[i] = new double*[num_nodes[i]];
		rprop_signs[i] = new char*[num_nodes[i]];
		for(int j = 0; j < num_nodes[i]; j++){
			this->rprop_updates[i][j] = new double[num_nodes[i + 1]];
			rprop_signs[i][j] = new char[num_nodes[i + 1]];
			for(int k = 0; k < num_nodes[i + 1]; k++){
				this->rprop_updates[i][j][k] = rprop_updates[i][j][k];
			}
		}
	}
	first_rprop = true;
}

void matrix_nn::update_p_keep(double *p_keep){
	if(this->p_keep != NULL){
		delete [] this->p_keep;
	}
	this->p_keep = new double[num_layers - 2];
	for(int i = 0; i < num_layers - 2; i++){
		this->p_keep[i] = p_keep[i];
	}
}

void matrix_nn::delete_keep_mask(){
	if(keep_mask[0] != NULL){
		for(int i = 0; i < num_layers - 2; i++){
			for(int j = 0; j < num_sets; j++){
				delete [] keep_mask[i][j];
			}
			delete [] keep_mask[i];
		}
		keep_mask[0] = NULL;
	}
}

void matrix_nn::gen_keep_mask(){
	for(int i = 0; i < num_layers - 2; i++){
		keep_mask[i] = new bool*[num_sets];
		for(int j = 0; j < num_sets; j++){
			keep_mask[i][j] = new bool[num_nodes[i + 1]];
			for(int k = 0; k < num_nodes[i + 1]; k++){
				double p = (rand() % 100) / 100.0;
				keep_mask[i][j][k] = p < p_keep[i];
			}
		}
	}
}

void matrix_nn::delete_forward_products(){	
	if(node_inputs[0] != NULL){
		for(int i = 0; i < num_layers - 1; i++){
			for(int j = 0; j < this->num_sets; j++){
				delete [] node_inputs[i][j];
				delete [] node_activations[i][j];
			}
			delete [] node_inputs[i];
			delete [] node_activations[i];
		}
		node_inputs[0] = NULL;
	}
}

void matrix_nn::feed_forward(){
	delete_forward_products();
	for(int i = 0; i < num_layers - 1; i++){
		node_inputs[i] = dot(i == 0 ? inputs : node_activations[i - 1], num_sets, num_nodes[i], weights[i], num_nodes[i], num_nodes[i + 1]);
		node_activations[i] = element_wise_activation_function(node_inputs[i], num_sets, num_nodes[i + 1], activation_functions[i]);
		if(i < num_layers - 2){
			for(int j = 0; j < num_sets; j++){
				for(int k = 0; k < num_nodes[i + 1]; k++){
					node_activations[i][j][k] = keep_mask[i][j][k] ? node_activations[i][j][k] : 0;
				}
			}
		}
	}
}

void matrix_nn::feed_forward_no_dropout(){
	delete_forward_products();
	for(int i = 0; i < num_layers - 1; i++){
		node_inputs[i] = dot(i == 0 ? inputs : node_activations[i - 1], num_sets, num_nodes[i], weights[i], num_nodes[i], num_nodes[i + 1]);
		node_activations[i] = element_wise_activation_function(node_inputs[i], num_sets, num_nodes[i + 1], activation_functions[i]);
		if(i < num_layers - 2){
			for(int j = 0; j < num_sets; j++){
				for(int k = 0; k < num_nodes[i + 1]; k++){
					node_activations[i][j][k] *= p_keep[i];
				}
			}
		}
	}
}

double **matrix_nn::calc_loss_matrix(double **last_node_gradients){
	double **loss_matrix = new double*[num_sets];
	for(int i = 0; i < num_sets; i++){
		loss_matrix[i] = new double[num_nodes[num_layers - 1]];
		for(int j = 0; j < num_nodes[num_layers - 1]; j++){
			loss_matrix[i][j] = (*m_loss_function)(last_node_gradients[i][j], node_activations[num_layers - 2][i][j], desired_activations[i][j]);
		}
	}
	return loss_matrix;
}

void matrix_nn::delete_backward_products(){
	if(input_gradients[0] != NULL){
		for(int i = 0; i < num_layers - 1; i++){
			for(int j = 0; j < num_sets; j++){
				delete [] input_gradients[i][j];
			}
			delete [] input_gradients[i];
		}
		input_gradients[0] = NULL;
	}
	if(activation_gradients[0] != NULL){
		for(int i = 0; i < num_layers - 2; i++){
			for(int j = 0; j < num_sets; j++){
				delete [] activation_gradients[i][j];
			}
			delete [] activation_gradients[i];
		}
		activation_gradients[0] = NULL;
	}
}

void matrix_nn::feed_backward(){
	delete_backward_products();
	for(int i = num_layers - 2; i >= 0; i--){
		input_gradients[i] = element_wise_activation_function(node_inputs[i], num_sets, num_nodes[i + 1], activation_functions[i]->get_derivative());
		double **temp = input_gradients[i];
		if(i == num_layers - 2){
			input_gradients[i] = calc_loss_matrix(input_gradients[i]);
		}else{
			input_gradients[i] = cross(input_gradients[i], num_sets, num_nodes[i + 1], activation_gradients[i], num_sets, num_nodes[i + 1]);
		}
		for(int j = 0; j < num_sets; j++){
			delete [] temp[j];
		}
		delete [] temp;

		if(i > 0){
			activation_gradients[i - 1] = second_inverse_dot(input_gradients[i], num_sets, num_nodes[i + 1], weights[i], num_nodes[i], num_nodes[i + 1]);
		}
	}
}

void matrix_nn::train_rprop(){	
	double ***weight_gradients = new double**[num_layers - 1];
	feed_forward();
	feed_backward();
	for(int i = num_layers - 2; i >= 0; i--){
		weight_gradients[i] = first_inverse_dot(i == 0 ? inputs : node_activations[i - 1], num_sets, num_nodes[i], input_gradients[i], num_sets, num_nodes[i + 1]);
		for(int j = 0; j < num_nodes[i]; j++){
			for(int k = 0; k < num_nodes[i + 1]; k++){
				if(!first_rprop && ((weight_gradients[i][j][k] > 0 && rprop_signs[i][j][k] != '+') || (weight_gradients[i][j][k] < 0 && rprop_signs[i][j][k] != '-'))){
					rprop_updates[i][j][k] *= 0.5;
				}
				rprop_signs[i][j][k] = weight_gradients[i][j][k] > 0 ? '+' : (weight_gradients[i][j][k] < 0 ? '-' : '0');
				weights[i][j][k] += weight_gradients[i][j][k] > 0 ? -rprop_updates[i][j][k] : (weight_gradients[i][j][k] < 0 ? rprop_updates[i][j][k] : 0);
			}
		}
	}
	first_rprop = false;

	for(int i = num_layers - 2; i >= 0; i--){
		for(int j = 0; j < num_nodes[i]; j++){
			delete [] weight_gradients[i][j];
		}
		delete [] weight_gradients[i];
	}
	delete [] weight_gradients;
}

void matrix_nn::train_rprop(double **inputs, double **desired_activations, int num_sets, double ***rprop_updates, int iterations){
	update_sets(inputs, desired_activations, num_sets);
	update_rprop_updates(rprop_updates);
	for(int i = 0; i < iterations; i++){
		train_rprop();
	}
}

void matrix_nn::train(){
	double ***weight_gradients = new double**[num_layers - 1];
	feed_forward();
	feed_backward();
	for(int i = num_layers - 2; i >= 0; i--){
		weight_gradients[i] = first_inverse_dot(i == 0 ? inputs : node_activations[i - 1], num_sets, num_nodes[i], input_gradients[i], num_sets, num_nodes[i + 1]);
		for(int j = 0; j < num_nodes[i]; j++){
			for(int k = 0; k < num_nodes[i + 1]; k++){
				weights[i][j][k] += weight_gradients[i][j][k] * (-training_rate);
			}
		}
	}

	for(int i = num_layers - 2; i >= 0; i--){
		for(int j = 0; j < num_nodes[i]; j++){
			delete [] weight_gradients[i][j];
		}
		delete [] weight_gradients[i];
	}
	delete [] weight_gradients;
}

void matrix_nn::train(double **inputs, double **desired_activations, int num_sets, int iterations){
	update_sets(inputs, desired_activations, num_sets);
	for(int i = 0; i < iterations; i++){
		train();
	}
}

double **matrix_nn::dot(double **m1, int r1, int c1, double **m2, int r2, int c2){
	double **res = NULL;
	if(r1 > 0 && c2 > 0){
		res = new double*[r1];
		for(int i = 0; i < r1; i++){
			res[i] = new double[c2];
		}

		for(int i = 0; i < r1; i++){
			for(int j = 0; j < c2; j++){
				double sum = 0;
				for(int k = 0; k < c1; k++){
					sum += m1[i][k] * m2[k][j];
				}
				res[i][j] = sum;
			}
		}
	}
	return res;
}

double **matrix_nn::element_wise_activation_function(double **m, int r, int c, activation_function *f){
	double **res = NULL;
	if(r > 0 && c > 0){
		res = new double*[r];
		for(int i = 0; i < r; i++){
			res[i] = new double[c];
			for(int j = 0; j < c; j++){
				res[i][j] = (*f)(m[i][j]);
			}
		}
	}
	return res;
}

double **matrix_nn::first_inverse_dot(double **m1, int r1, int c1, double **m2, int r2, int c2){
	double **res = NULL;
	if(c1 > 0 && c2 > 0){
		res = new double*[c1];
		for(int i = 0; i < c1; i++){
			res[i] = new double[c2];
			for(int j = 0; j < c2; j++){
				double sum = 0;
				for(int k = 0; k < r1; k++){
					sum += m1[k][i] * m2[k][j];
				}
				res[i][j] = sum;
			}
		}
	}
	return res;
}

double **matrix_nn::second_inverse_dot(double **m1, int r1, int c1, double **m2, int r2, int c2){
	double **res = NULL;
	if(r1 > 0 && r2 > 0){
		res = new double*[r1];
		for(int i = 0; i < r1; i++){
			res[i] = new double[r2];
			for(int j = 0; j < r2; j++){
				double sum = 0;
				for(int k = 0; k < c1; k++){
					sum += m1[i][k] * m2[j][k];
				}
				res[i][j] = sum;
			}
		}
	}
	return res;
}

double **matrix_nn::cross(double **m1, int r1, int c1, double **m2, int r2, int c2){
	double **res = NULL;
	if(r1 > 0 && c1 > 0){
		res = new double*[r1];
		for(int i = 0; i < r1; i++){
			res[i] = new double[c1];
			for(int j = 0; j < c1; j++){
				res[i][j] = m1[i][j] * m2[i][j];
			}
		}

	}
	return res;
}

double matrix_nn::get_output(int set_index, int node_index){
	return node_activations[num_layers - 2][set_index][node_index];
}

void matrix_nn::calc_outputs(double **inputs, int num_sets){
	update_num_sets(num_sets);
	update_inputs(inputs);
	feed_forward_no_dropout();
	if(activation_functions[num_layers - 2]->get_finalizer() != NULL){
		for(int i = 0; i < num_sets; i++){
			(*(activation_functions[num_layers - 2]->get_finalizer()))(node_activations[num_layers - 2][i], num_nodes[num_layers - 1]);
		}
	}
}

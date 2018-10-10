#include <cstdlib>
#include <ctime>
#include <iostream>
#include "mnist_iterator.h"
#include "matrix_nn.h"
#include "softmax.h"
#include "relu.h"
#include "mse.h"

using std::cout;

/**

	TESTS:
	Relu with 100,000 epochs: 99.98% accurate
	Sigmoid with 1,000,000 epochs: 99.97% accurate

	So relu became more accurate with a tenth the epochs...

	Then, tested relu with RProp. In 10,000 epochs, it was 99.94% accurate with backprop,
	and with RProp, error is essentially zero. We're talking 10^-9 and 10^-155 errors here.
	And that's with just 10,000 epochs. That's more accurate than backprop @ 100,000 epochs,
	as well as backprop w/ sigmoid @ 1,000,000 epochs. Speed has been increased by over 10,000%

	Keep in mind that rprop may require very specific initial update values and weight
	configurations. I had to mutiply by the layer plus one, squared, with a coefficient of 0.01.
	Any lower, it won't be accurate. Any higher, it will get stuck at a local minimum. Higher power,
	it will also get stuck.

*/
/*
int main(){
	srand(time(NULL));
	int num_layers = 3;
	int num_nodes[] = {2, 4, 2};
	softmax s;
	relu r;
	activation_function *functions[] = {&r, &s};
	mse m_mse;
	double *p_keep = new double[num_layers - 2];
	for(int i = 0; i < num_layers - 2; i++){
		p_keep[i] = 1;
	}
	matrix_nn m(num_layers, num_nodes, functions, &m_mse, .01, p_keep);
	double **inputs = new double*[2];
	inputs[0] = new double[2];
	inputs[0][0] = 1;
	inputs[0][1] = 1;
	inputs[1] = new double[2];
	inputs[1][0] = 1;
	inputs[1][1] = 0;
	double **desired_activations = new double*[2];
	desired_activations[0] = new double[2];
	desired_activations[0][0] = 1;
	desired_activations[0][1] = 0;
	desired_activations[1] = new double[2];
	desired_activations[1][0] = 0;
	desired_activations[1][1] = 1;
	m.calc_outputs(inputs, 2);
	cout << "1 XOR 1: " << (m.get_output(0, 0) * 100) << "% vs " << (m.get_output(0, 1) * 100) << "%\n";
	cout << "1 XOR 0: " << (m.get_output(1, 0) * 100) << "% vs " << (m.get_output(1, 1) * 100) << "%\n";
	double ***rprop_updates = new double**[num_layers - 1];
	for(int i = 0; i < num_layers - 1; i++){
		rprop_updates[i] = new double*[num_nodes[i]];
		for(int j = 0; j < num_nodes[i]; j++){
			rprop_updates[i][j] = new double[num_nodes[i + 1]];
			for(int k = 0; k < num_nodes[i + 1]; k++){
				rprop_updates[i][j][k] = 0.005 * (i + 1);
			}
		}
	}
	//m.train_rprop(inputs, desired_activations, 2, rprop_updates, 10000);
	m.train(inputs, desired_activations, 2, 100000);
	m.calc_outputs(inputs, 2);
	cout << "1 XOR 1: " << (m.get_output(0, 0) * 100) << "% vs " << (m.get_output(0, 1) * 100) << "%\n";
	cout << "1 XOR 0: " << (m.get_output(1, 0) * 100) << "% vs " << (m.get_output(1, 1) * 100) << "%\n";
	
	for(int i = 0; i < num_layers - 1; i++){
		for(int j = 0; j < num_nodes[i]; j++){
			delete [] rprop_updates[i][j];
		}
		delete [] rprop_updates[i];
	}
	delete [] p_keep;
	delete [] rprop_updates;
	delete [] desired_activations[0];
	delete [] desired_activations[1];
	delete [] desired_activations;
	delete [] inputs[0];
	delete [] inputs[1];
	delete [] inputs;
	return 0;
}*/

/*
int main(){
	srand(time(NULL));
	int num_layers = 3;
	int num_nodes[] = {784, 100, 10};
	int num_sets = 100;
	softmax s;
	relu r;
	activation_function *functions[] = {&r, &s};
	mse m_mse;
	double *p_keep = new double[num_layers - 2];
	for(int i = 0; i < num_layers - 2; i++){
		p_keep[i] = 1;
	}
	matrix_nn m(num_layers, num_nodes, functions, &m_mse, 5, p_keep);
	double **inputs = new double*[num_sets];
	double **desired_activations = new double*[num_sets];
	for(int i = 0; i < num_sets; i++){
		inputs[i] = new double[num_nodes[0]];
		desired_activations[i] = new double[num_nodes[num_layers - 1]];
	}
	m.calc_outputs(inputs, num_sets);
	cout << "1 XOR 1: " << (m.get_output(0, 0) * 100) << "% vs " << (m.get_output(0, 1) * 100) << "%\n";
	cout << "1 XOR 0: " << (m.get_output(1, 0) * 100) << "% vs " << (m.get_output(1, 1) * 100) << "%\n";
	double ***rprop_updates = new double**[num_layers - 1];
	for(int i = 0; i < num_layers - 1; i++){
		rprop_updates[i] = new double*[num_nodes[i]];
		for(int j = 0; j < num_nodes[i]; j++){
			rprop_updates[i][j] = new double[num_nodes[i + 1]];
			for(int k = 0; k < num_nodes[i + 1]; k++){
				rprop_updates[i][j][k] = 0.01 * (i + 1) * (i + 1);
			}
		}
	}
	//m.train_rprop(inputs, desired_activations, 2, rprop_updates, 10000);
	m.train(inputs, desired_activations, num_sets, 5);
	m.calc_outputs(inputs, num_sets);
	cout << "1 XOR 1: " << (m.get_output(0, 0) * 100) << "% vs " << (m.get_output(0, 1) * 100) << "%\n";
	cout << "1 XOR 0: " << (m.get_output(1, 0) * 100) << "% vs " << (m.get_output(1, 1) * 100) << "%\n";
	
	for(int i = 0; i < num_layers - 1; i++){
		for(int j = 0; j < num_nodes[i]; j++){
			delete [] rprop_updates[i][j];
		}
		delete [] rprop_updates[i];
	}

	delete [] p_keep;
	delete [] rprop_updates;
	
	for(int i = 0; i < num_sets; i++){
		delete [] inputs[i];
		delete [] desired_activations[i];
	}
	delete [] inputs;
	delete [] desired_activations;
	return 0;
}*/

int main(){
	int num_layers = 3;
	int num_nodes[] = {28*28, 150, 10};
	relu r;
	softmax s;
	activation_function *functions[] = {&r, &s};
	mse m;
	double p_keep[] = {1.0};
	double backprop_training_rate = 0.01;
	matrix_nn nn(num_layers, num_nodes, functions, &m, backprop_training_rate, p_keep);
	mnist_iterator itr;

	int num_sets = 60000;
	int read_size = 1000;
	int batch_size = 50;
	int num_reads = (num_sets / read_size) * read_size < num_sets ? (num_sets / read_size) + 1 : (num_sets / read_size);

	for(int i = 0; i < num_reads; i++){
		int remaining = num_sets - i * read_size;
		int specific_read_size = read_size < remaining ? read_size : remaining;
		double **batch = itr.get_inputs(i * read_size, specific_read_size, mnist_iterator::TYPE_TRAIN);
		double **desired_outputs = itr.get_desired_outputs(i * read_size, specific_read_size, mnist_iterator::TYPE_TRAIN);
		cout << i << "\n";

		int num_batches = (specific_read_size / batch_size) * batch_size < specific_read_size ? (specific_read_size / batch_size) + 1 : (specific_read_size / batch_size);
		for(int j = 0; j < num_batches; j++){
			remaining = specific_read_size - j * batch_size;
			double specific_batch_size = batch_size < remaining ? batch_size : remaining;
			nn.train(batch + j * batch_size, desired_outputs + j * batch_size, specific_batch_size, 1);
		}
		for(int j = 0; j < specific_read_size; j++){
			delete [] batch[j];
			delete [] desired_outputs[j];
		}
		delete [] batch;
		delete [] desired_outputs;
	}

	num_sets = 10000;
	read_size = 1000;
	batch_size = 50;
	num_reads = (num_sets / read_size) * read_size < num_sets ? (num_sets / read_size) + 1 : (num_sets / read_size);

	int num_correct = 0;

	for(int i = 0; i < num_reads; i++){
		int remaining = num_sets - i * read_size;
		int specific_read_size = read_size < remaining ? read_size : remaining;
		double **batch = itr.get_inputs(i * read_size, specific_read_size, mnist_iterator::TYPE_TEST);
		double **desired_outputs = itr.get_desired_outputs(i * read_size, specific_read_size, mnist_iterator::TYPE_TEST);

		int num_batches = (specific_read_size / batch_size) * batch_size < specific_read_size ? (specific_read_size / batch_size) + 1 : (specific_read_size / batch_size);
		for(int j = 0; j < num_batches; j++){
			remaining = specific_read_size - j * batch_size;
			double specific_batch_size = batch_size < remaining ? batch_size : remaining;
			nn.calc_outputs(batch + j * batch_size, specific_batch_size);
			for(int k = 0; k < specific_batch_size; k++){
				int highest_predicted_index = 0, highest_actual_index = 0;
				double highest_predicted = nn.get_output(k, 0), highest_actual = (desired_outputs + j * batch_size)[k][0];
				for(int l = 1; l < 10; l++){
					double output = nn.get_output(k, l);
					if(output > highest_predicted){
						highest_predicted_index = l;
						highest_predicted = output;
					}
					double desired_output = (desired_outputs + j * batch_size)[k][l];
					if(desired_output > highest_actual){
						highest_actual_index = l;
						highest_actual = desired_output;
					}
				}
				
				if(highest_predicted_index == highest_actual_index)
					num_correct++;
			}
		}

		for(int j = 0; j < specific_read_size; j++){
			delete [] batch[j];
			delete [] desired_outputs[j];
		}
		delete [] batch;
		delete [] desired_outputs;
	}

	cout << "Percent correct: " << (double) num_correct / 10000.0 * 100.0 << "\n";
	
	return 0;
}
